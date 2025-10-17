#pragma once
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "dit_linear.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
// VAE model compatible with huggingface weights
//  ref to:
//  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py

namespace xllm {
class VAEImageProcessorImpl : public torch::nn::Module {
 public:
  VAEImageProcessorImpl(ModelContext context) {
    const auto& model_args = context.get_model_args();
    scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
  }

  std::pair<int64_t, int64_t> adjust_dimensions(int64_t height,
                                                int64_t width) const {
    height = height - (height % scale_factor_);
    width = width - (width % scale_factor_);
    return {height, width};
  }

  torch::Tensor preprocess(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt,
      const std::string& resize_mode = "default",
      std::optional<std::tuple<int64_t, int64_t, int64_t, int64_t>>
          crop_coords = std::nullopt) {
    torch::Tensor processed = image.clone();
    if (processed.dtype() != torch::kFloat32) {
      processed = processed.to(torch::kFloat32);
    }
    if (processed.max().item<float>() > 1.1f) {
      processed = processed / 255.0f;
    }
    if (crop_coords.has_value()) {
      auto [x1, y1, x2, y2] = crop_coords.value();
      x1 = std::max(int64_t(0), x1);
      y1 = std::max(int64_t(0), y1);
      x2 = std::min(processed.size(-1), x2);
      y2 = std::min(processed.size(-2), y2);

      if (processed.dim() == 3) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      } else if (processed.dim() == 4) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      }
    }
    if (do_convert_grayscale_ && processed.size(1) == 3) {
      std::vector<float> weights = {0.299f, 0.587f, 0.114f};
      torch::Tensor weight_tensor =
          torch::tensor(weights, torch::kFloat32).view({1, 3, 1, 1});
      if (processed.dim() == 3) {
        weight_tensor = weight_tensor.squeeze(0);
      }
      processed = torch::sum(processed * weight_tensor, 1, true);
    } else if (do_convert_rgb_ && processed.size(1) == 1) {
      processed = torch::cat({processed, processed, processed}, 1);
    }

    auto [target_h, target_w] =
        get_default_height_width(processed, height, width);
    if (do_resize_) {
      processed = resize(processed, target_h, target_w);
    }

    if (do_normalize_) {
      processed = normalize(processed);
    }
    if (do_binarize_) {
      processed = (processed >= 0.5f).to(torch::kFloat32);
    }

    return processed;
  }

  torch::Tensor postprocess(
      const torch::Tensor& tensor,
      const std::string& output_type = "pt",
      std::optional<std::vector<bool>> do_denormalize = std::nullopt) {
    torch::Tensor processed = tensor.clone();
    if (do_normalize_) {
      if (!do_denormalize.has_value()) {
        processed = denormalize(processed);
      } else {
        for (int64_t i = 0; i < processed.size(0); ++i) {
          if (i < do_denormalize.value().size() && do_denormalize.value()[i]) {
            processed[i] = denormalize(processed[i]);
          }
        }
      }
    }
    if (output_type == "np") {
      return processed.permute({0, 2, 3, 1}).contiguous();
    }
    return processed;
  }

 private:
  std::pair<int64_t, int64_t> get_default_height_width(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt) const {
    int64_t h, w;
    if (image.dim() == 3) {
      h = image.size(1);
      w = image.size(2);
    } else if (image.dim() == 4) {
      h = image.size(2);
      w = image.size(3);
    } else {
      throw std::invalid_argument("Invalid image tensor dimensions");
    }

    int64_t target_h = height.value_or(h);
    int64_t target_w = width.value_or(w);
    return adjust_dimensions(target_h, target_w);
  }

  torch::Tensor normalize(const torch::Tensor& tensor) const {
    return 2.0 * tensor - 1.0;
  }

  torch::Tensor denormalize(const torch::Tensor& tensor) const {
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0);
  }

  torch::Tensor resize(const torch::Tensor& image,
                       int64_t target_height,
                       int64_t target_width) const {
    return torch::nn::functional::interpolate(
        image,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_height, target_width})
            .align_corners(false));
  }

 private:
  int scale_factor_ = 8;
  bool do_resize_ = true;
  bool do_normalize_ = true;
  bool do_binarize_ = false;
  bool do_convert_rgb_ = false;
  bool do_convert_grayscale_ = false;
};
TORCH_MODULE(VAEImageProcessor);

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(ModelContext context) {
    ModelArgs model_args = context.get_model_args();
    int64_t query_dim = model_args.block_out_channels().back();
    int64_t head_dim = model_args.block_out_channels().back();
    int64_t num_heads = query_dim / head_dim;
    num_heads_ = num_heads;
    int64_t norm_num_groups = model_args.norm_num_groups();
    if (norm_num_groups > 0) {
      group_norm_ =
          register_module("group_norm",
                          torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                   norm_num_groups, query_dim)
                                                   .eps(1e-6f)));
    }
    int64_t inner_dim = head_dim * num_heads;
    to_q_ = register_module("to_q", DiTLinear(query_dim, inner_dim, true));
    to_k_ = register_module("to_k", DiTLinear(query_dim, inner_dim, true));
    to_v_ = register_module("to_v", DiTLinear(query_dim, inner_dim, true));
    to_out_ = register_module("to_out", DiTLinear(inner_dim, query_dim, true));
  }

  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor temb) {
    torch::Tensor residual = hidden_states;
    int64_t input_ndim = hidden_states.dim();
    int64_t batch_size;
    int64_t channel;
    int64_t height;
    int64_t width;
    if (input_ndim == 4) {
      batch_size = hidden_states.size(0);
      channel = hidden_states.size(1);
      height = hidden_states.size(2);
      width = hidden_states.size(3);

      // reshape tensor：[B, C, H, W] -> [B, C, H*W]
      hidden_states = hidden_states.view({batch_size, channel, height * width});

      // swap dim 1&2：[B, C, H*W] -> [B, H*W, C]
      hidden_states = hidden_states.transpose(1, 2);
    }
    if (temb.defined()) {
      torch::IntArrayRef input_shape = temb.sizes();
      batch_size = input_shape[0];
      int64_t seq_length = input_shape[1];
    }
    hidden_states =
        group_norm_->forward(hidden_states.transpose(1, 2)).transpose(1, 2);
    torch::Tensor query = to_q_(hidden_states);
    // if temb is none
    if (!temb.defined()) {
      temb = hidden_states.clone();
    }
    torch::Tensor key = to_k_(temb);
    torch::Tensor value = to_v_(temb);
    int64_t inner_dim = key.size(-1);
    int64_t head_dim = inner_dim / num_heads_;
    query = query.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    key = key.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    value = value.view({batch_size, -1, num_heads_, head_dim}).transpose(1, 2);
    hidden_states = at::scaled_dot_product_attention(query,
                                                     key,
                                                     value,
                                                     c10::nullopt,  // attn_mask
                                                     0.0,           // dropout_p
                                                     false          // is_causal
    );
    hidden_states = hidden_states.transpose(1, 2).reshape(
        {batch_size, -1, num_heads_ * head_dim});
    hidden_states = hidden_states.to(query.dtype());
    hidden_states = to_out_(hidden_states);
    // alignment right
    if (input_ndim == 4) {
      hidden_states = hidden_states.transpose(-1, -2).reshape(
          {batch_size, channel, height, width});
    }
    hidden_states = hidden_states + residual;
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    // to_q_
    state_dict.copy_tensor_to("to_q.weight", to_q_->weight);
    state_dict.copy_tensor_to("to_q.bias", to_q_->bias);
    // to_k_
    state_dict.copy_tensor_to("to_k.weight", to_k_->weight);
    state_dict.copy_tensor_to("to_k.bias", to_k_->bias);
    // to_v_
    state_dict.copy_tensor_to("to_v.weight", to_v_->weight);
    state_dict.copy_tensor_to("to_v.bias", to_v_->bias);
    // to_out_
    state_dict.copy_tensor_to("to_out.0.weight", to_out_->weight);
    state_dict.copy_tensor_to("to_out.0.bias", to_out_->bias);
    if (group_norm_) {
      state_dict.copy_tensor_to("group_norm.weight", group_norm_->weight);
      state_dict.copy_tensor_to("group_norm.bias", group_norm_->bias);
    }
  }

 private:
  torch::nn::GroupNorm group_norm_{nullptr};
  DiTLinear to_q_{nullptr};
  DiTLinear to_k_{nullptr};
  DiTLinear to_v_{nullptr};
  DiTLinear to_out_{nullptr};
  int64_t num_heads_;
};
TORCH_MODULE(Attention);

class Downsample2DImpl : public torch::nn::Module {
 public:
  Downsample2DImpl(int64_t channels, int64_t padding = 1) {
    padding_ = padding;
    conv_ = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .stride(2)
                              .padding(padding)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const std::vector<torch::Tensor>& args = {}) {
    torch::Tensor x = hidden_states;
    if (padding_ == 0) {
      x = torch::nn::functional::pad(
          x,
          torch::nn::functional::PadFuncOptions({0, 1, 0, 1})
              .mode(torch::kConstant)
              .value(0.0f));
    }
    x = conv_(x);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    state_dict.copy_tensor_to("conv.weight", conv_->weight);
    state_dict.copy_tensor_to("conv.bias", conv_->bias);
  }

 private:
  int64_t padding_;
  torch::nn::Conv2d conv_{nullptr};
};
TORCH_MODULE(Downsample2D);

class Upsample2DImpl : public torch::nn::Module {
 public:
  Upsample2DImpl(int64_t channels) {
    conv_ = register_module(
        "conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .padding(1)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const std::vector<int64_t>& output_size = {}) {
    torch::Tensor x = hidden_states;
    if (x.size(0) >= 64) {
      x = x.contiguous();
    }

    torch::nn::functional::InterpolateFuncOptions opts;
    opts.mode(torch::kNearest);

    if (!output_size.empty()) {
      opts.size(output_size);
    } else {
      opts.scale_factor(std::vector<double>{2.0, 2.0});
    }

    x = torch::nn::functional::interpolate(x, opts);
    x = conv_(x);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    state_dict.copy_tensor_to("conv.weight", conv_->weight);
    state_dict.copy_tensor_to("conv.bias", conv_->bias);
  }

 private:
  torch::nn::ConvTranspose2d conv_transpose_ = {nullptr};
  torch::nn::Conv2d conv_{nullptr};
};
TORCH_MODULE(Upsample2D);

class ResnetBlock2DImpl : public torch::nn::Module {
 public:
  ResnetBlock2DImpl(ModelContext context,
                    int64_t in_channels,
                    int64_t out_channels,
                    const std::string& time_embedding_norm = "default")
      : in_channels_(in_channels),
        out_channels_(out_channels),
        time_embedding_norm_(time_embedding_norm) {
    ModelArgs model_args = context.get_model_args();
    int64_t groups_out_val = model_args.norm_num_groups();

    norm1_ =
        register_module("norm1",
                        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                 groups_out_val, in_channels_)
                                                 .eps(1e-6f)));
    conv1_ =
        register_module("conv1",
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                              in_channels_, out_channels_, 3)
                                              .stride(1)
                                              .padding(1)
                                              .bias(true)));

    norm2_ =
        register_module("norm2",
                        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                                 groups_out_val,  // num_groups
                                                 out_channels_)  // num_channels
                                                 .eps(1e-6f)));

    int64_t conv2_out_channels = out_channels_;
    conv2_ = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels_,
                                                   conv2_out_channels,
                                                   3  // kernel_size=3
                                                   )
                              .stride(1)
                              .padding(1)
                              .bias(true)));

    nonlinearity_ =
        register_module("nonlinearity", torch::nn::Functional(torch::silu));

    bool use_in_shortcut_val = in_channels_ != conv2_out_channels;
    if (use_in_shortcut_val) {
      conv_shortcut_ = register_module(
          "conv_shortcut",
          torch::nn::Conv2d(
              torch::nn::Conv2dOptions(in_channels_, conv2_out_channels, 1)
                  .stride(1)
                  .padding(0)
                  .bias(true)));
    }
  }

  torch::Tensor forward(const torch::Tensor& input_tensor,
                        const torch::Tensor& temb = torch::Tensor()) {
    torch::Tensor hidden_states = input_tensor;
    hidden_states = norm1_(hidden_states);
    hidden_states = nonlinearity_(hidden_states);
    hidden_states = conv1_(hidden_states);
    torch::Tensor temb_processed = temb;
    torch::Tensor input_tensor_processed = input_tensor;

    if (time_embedding_norm_ == "default") {
      if (temb_processed.defined()) {
        hidden_states = hidden_states + temb_processed;
      }
      hidden_states = norm2_(hidden_states);
    } else if (time_embedding_norm_ == "scale_shift") {
      if (!temb_processed.defined()) {
        throw std::invalid_argument(
            "temb must be defined for scale_shift time_embedding_norm");
      }
      auto chunks = torch::chunk(temb_processed, 2, 1);
      torch::Tensor time_scale = chunks[0];
      torch::Tensor time_shift = chunks[1];
      hidden_states = norm2_(hidden_states);
      hidden_states = hidden_states * (1 + time_scale) + time_shift;
    } else {
      hidden_states = norm2_(hidden_states);
    }
    hidden_states = nonlinearity_(hidden_states);
    hidden_states = conv2_(hidden_states);

    if (conv_shortcut_) {
      input_tensor_processed =
          conv_shortcut_(input_tensor_processed.contiguous());
    }
    torch::Tensor output_tensor = (input_tensor_processed + hidden_states);

    return output_tensor;
  }

  void load_state_dict(const StateDict& state_dict) {
    // conv1_
    state_dict.copy_tensor_to("conv1.weight", conv1_->weight);
    state_dict.copy_tensor_to("conv1.bias", conv1_->bias);
    // norm1_
    state_dict.copy_tensor_to("norm1.weight", norm1_->weight);
    state_dict.copy_tensor_to("norm1.bias", norm1_->bias);
    // norm2_
    state_dict.copy_tensor_to("norm2.weight", norm2_->weight);
    state_dict.copy_tensor_to("norm2.bias", norm2_->bias);
    // conv2_
    state_dict.copy_tensor_to("conv2.weight", conv2_->weight);
    state_dict.copy_tensor_to("conv2.bias", conv2_->bias);
    if (conv_shortcut_) {
      // conv_shortcut_
      state_dict.copy_tensor_to("conv_shortcut.weight", conv_shortcut_->weight);
      state_dict.copy_tensor_to("conv_shortcut.bias", conv_shortcut_->bias);
    }
  }

 private:
  int64_t in_channels_;
  int64_t out_channels_;
  std::string time_embedding_norm_;

  torch::nn::GroupNorm norm1_{nullptr};
  torch::nn::Conv2d conv1_{nullptr};
  torch::nn::GroupNorm norm2_{nullptr};
  torch::nn::Conv2d conv2_{nullptr};
  torch::nn::Functional nonlinearity_{nullptr};
  torch::nn::Conv2d conv_shortcut_{nullptr};
};
TORCH_MODULE(ResnetBlock2D);

class UNetMidBlock2DImpl : public torch::nn::Module {
 public:
  UNetMidBlock2DImpl(ModelContext context) {
    ModelArgs model_args = context.get_model_args();
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    attentions_ = register_module("attentions", torch::nn::ModuleList());
    int64_t in_channels = model_args.block_out_channels().back();
    add_attention_ = model_args.mid_block_add_attention();
    resnets_->push_back(
        ResnetBlock2D(context, in_channels, in_channels, "default"));
    int64_t attn_head_dim = model_args.block_out_channels().back();
    int64_t num_heads = in_channels / attn_head_dim;
    if (add_attention_) {
      attentions_->push_back(Attention(context));
    } else {
      // Add an empty module as a placeholder.
      attentions_->push_back(torch::nn::Sequential());
    }
    resnets_->push_back(
        ResnetBlock2D(context, in_channels, in_channels, "default"));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor()) {
    torch::Tensor current_hidden =
        resnets_[0]->as<ResnetBlock2D>()->forward(hidden_states, temb);
    for (size_t i = 0; i < attentions_->size(); ++i) {
      if (add_attention_) {
        auto attn = attentions_[i]->as<Attention>();
        current_hidden = attn->forward(current_hidden, temb);
      }
      auto resnet = resnets_[i + 1]->as<ResnetBlock2D>();
      current_hidden = resnet->forward(current_hidden, temb);
    }

    return current_hidden;
  }
  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<ResnetBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    for (size_t i = 0; i < attentions_->size(); ++i) {
      if (add_attention_) {
        attentions_[i]->as<Attention>()->load_state_dict(
            state_dict.get_dict_with_prefix("attentions." + std::to_string(i) +
                                            "."));
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList attentions_{nullptr};
  bool add_attention_;
};
TORCH_MODULE(UNetMidBlock2D);

class DownEncoderBlock2DImpl : public torch::nn::Module {
 public:
  DownEncoderBlock2DImpl(ModelContext context,
                         int64_t in_channels,
                         int64_t out_channels,
                         bool add_downsample = true) {
    ModelArgs model_args = context.get_model_args();
    int64_t num_layers = model_args.layers_per_block();
    resnets_ = register_module("resnets", torch::nn::ModuleList());
    downsamplers_ = register_module("downsamplers", torch::nn::ModuleList());
    // initialize resnet blocks
    for (int64_t i = 0; i < num_layers; ++i) {
      const int64_t current_in_channels = (i == 0) ? in_channels : out_channels;

      resnets_->push_back(ResnetBlock2D(context,
                                        current_in_channels,  // in channels
                                        out_channels,
                                        "default"));
    }

    // initialize downsamplers if needed
    if (add_downsample) {
      downsamplers_->push_back(Downsample2D(out_channels, 0));
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor(),
                        const std::vector<torch::Tensor>& args = {}) {
    std::vector<torch::Tensor> output_states;
    torch::Tensor current_hidden = hidden_states;
    // handle resnet blocks
    for (size_t i = 0; i < resnets_->size(); ++i) {
      auto resnet = resnets_[i]->as<ResnetBlock2D>();
      current_hidden =
          resnet->forward(current_hidden.clone(),
                          temb.defined() ? temb.clone() : torch::Tensor());
      output_states.push_back(current_hidden);
    }
    // handle downsampling
    if (downsamplers_) {
      for (size_t i = 0; i < downsamplers_->size(); ++i) {
        auto downsampler = downsamplers_[i]->as<Downsample2D>();
        current_hidden = downsampler->forward(current_hidden.clone());
        output_states.push_back(current_hidden);
      }
    }
    return current_hidden;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<ResnetBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }
    for (size_t i = 0; i < downsamplers_->size(); ++i) {
      downsamplers_[i]->as<Downsample2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("downsamplers." + std::to_string(i) +
                                          "."));
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList downsamplers_{nullptr};
};
TORCH_MODULE(DownEncoderBlock2D);

class UpDecoderBlock2DImpl : public torch::nn::Module {
 public:
  UpDecoderBlock2DImpl(ModelContext context,
                       int64_t in_channels,
                       int64_t out_channels,
                       bool add_upsample = true) {
    ModelArgs model_args = context.get_model_args();
    int64_t num_layers = model_args.layers_per_block() + 1;
    resnets_ = register_module("resnets", torch::nn::ModuleList());

    for (int64_t i = 0; i < num_layers; ++i) {
      int64_t input_channels = (i == 0) ? in_channels : out_channels;
      resnets_->push_back(
          ResnetBlock2D(context, input_channels, out_channels, "group"));
    }

    if (add_upsample) {
      add_upsample_ = true;
      upsamplers_ = register_module("upsamplers", torch::nn::ModuleList());
      upsamplers_->push_back(Upsample2D(out_channels));
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb = torch::Tensor(),
                        const std::vector<torch::Tensor>& skip_states = {},
                        const std::vector<torch::Tensor>& args = {}) {
    torch::Tensor current_hidden = hidden_states;

    for (size_t i = 0; i < resnets_->size(); ++i) {
      current_hidden =
          resnets_[i]->as<ResnetBlock2D>()->forward(current_hidden, temb);
    }
    if (add_upsample_) {
      for (size_t i = 0; i < upsamplers_->size(); ++i) {
        auto* upsampler = upsamplers_->children()[i]->as<Upsample2D>();
        current_hidden = upsampler->forward(current_hidden);
      }
    }
    return current_hidden;
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < resnets_->size(); ++i) {
      resnets_[i]->as<ResnetBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("resnets." + std::to_string(i) +
                                          "."));
    }

    if (add_upsample_) {
      for (size_t i = 0; i < upsamplers_->size(); ++i) {
        upsamplers_[i]->as<Upsample2D>()->load_state_dict(
            state_dict.get_dict_with_prefix("upsamplers." + std::to_string(i) +
                                            "."));
      }
    }
  }

 private:
  torch::nn::ModuleList resnets_{nullptr};
  torch::nn::ModuleList upsamplers_{nullptr};
  bool add_upsample_ = false;
};
TORCH_MODULE(UpDecoderBlock2D);

// VAE standard encoder implementation
// This class is used to encode images into latent representations.
class VAEEncoderImpl : public torch::nn::Module {
 public:
  VAEEncoderImpl(const ModelContext& context) {
    ModelArgs args = context.get_model_args();
    down_blocks_ = register_module("down_blocks", torch::nn::ModuleList());
    conv_in_ = register_module(
        "conv_in",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(args.in_channels(),
                                                   args.block_out_channels()[0],
                                                   3)
                              .stride(1)
                              .padding(1)
                              .bias(true)));
    // downblocks
    int32_t output_channels = args.block_out_channels()[0];
    for (size_t i = 0; i < args.down_block_types().size(); i++) {
      int32_t input_channels = output_channels;
      output_channels = args.block_out_channels()[i];
      bool is_final_block = (i == args.block_out_channels().size() - 1);
      auto down_block = DownEncoderBlock2D(
          context, input_channels, output_channels, !is_final_block);
      down_blocks_->push_back(down_block);
    }
    // mid blocks
    mid_block_ = register_module("mid_block", UNetMidBlock2D(context));
    conv_norm_out_ = register_module(
        "conv_norm_out",
        torch::nn::GroupNorm(
            torch::nn::GroupNormOptions(args.norm_num_groups(),
                                        args.block_out_channels().back())
                .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                args.block_out_channels().back(), 2 * args.latent_channels(), 3)
                .padding(1)
                .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& images) {
    auto sample = conv_in_(images);
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      sample = down_blocks_[i]->as<DownEncoderBlock2D>()->forward(sample);
    }
    sample = mid_block_(sample);

    sample = conv_norm_out_(sample);

    sample = conv_act_(sample);

    sample = conv_out_(sample);
    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    LOG(INFO) << "Loading state_dict for VAEEecoder";
    // conv_in_
    state_dict.copy_tensor_to("conv_in.weight", conv_in_->weight);
    state_dict.copy_tensor_to("conv_in.bias", conv_in_->bias);
    // conv_norm_out_
    state_dict.copy_tensor_to("conv_norm_out.weight", conv_norm_out_->weight);
    state_dict.copy_tensor_to("conv_norm_out.bias", conv_norm_out_->bias);
    // conv_out_
    state_dict.copy_tensor_to("conv_out.weight", conv_out_->weight);
    state_dict.copy_tensor_to("conv_out.bias", conv_out_->bias);
    for (size_t i = 0; i < down_blocks_->size(); ++i) {
      down_blocks_[i]->as<DownEncoderBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("down_blocks." + std::to_string(i) +
                                          "."));
    }
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
  }

 private:
  torch::nn::Conv2d conv_in_{nullptr};
  torch::nn::Conv2d conv_out_{nullptr};
  torch::nn::Functional conv_act_{nullptr};
  torch::nn::GroupNorm conv_norm_out_{nullptr};
  torch::nn::ModuleList down_blocks_{nullptr};
  UNetMidBlock2D mid_block_{nullptr};
};
TORCH_MODULE(VAEEncoder);

// VAE standart decoder implementation
//  This class is used to decode the latent representations into images.
class VAEDecoderImpl : public torch::nn::Module {
 public:
  VAEDecoderImpl(const ModelContext& context) {
    ModelArgs args = context.get_model_args();
    up_blocks_ = register_module("up_blocks", torch::nn::ModuleList());
    conv_in_ = register_module(
        "conv_in",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                args.latent_channels(), args.block_out_channels().back(), 3)
                .stride(1)
                .padding(1)
                .bias(true)));
    // mid blocks
    mid_block_ = register_module("mid_block", UNetMidBlock2D(context));
    // up blocks
    std::vector<int64_t> reversed_block_out_channels(
        args.block_out_channels().rbegin(), args.block_out_channels().rend());
    int64_t output_channel = reversed_block_out_channels[0];

    for (size_t i = 0; i < args.up_block_types().size(); ++i) {
      const std::string& up_block_type = args.up_block_types()[i];
      int64_t prev_output_channel = output_channel;
      output_channel = reversed_block_out_channels[i];
      bool is_final_block = (i == args.block_out_channels().size() - 1);
      // Create the up block using the factory function
      auto up_block = UpDecoderBlock2D(
          context, prev_output_channel, output_channel, !is_final_block);
      up_blocks_->push_back(
          register_module("up_block_" + std::to_string(i), up_block));
      prev_output_channel = output_channel;
    }
    conv_norm_out_ = register_module(
        "conv_norm_out",
        torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                                 args.norm_num_groups(),       // num_groups
                                 args.block_out_channels()[0]  // num_channels
                                 )
                                 .eps(1e-6)));
    conv_act_ = register_module("conv_act", torch::nn::Functional(torch::silu));
    conv_out_ = register_module(
        "conv_out",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(args.block_out_channels()[0],
                                                   args.out_channels(),
                                                   3)
                              .padding(1)
                              .bias(true)));
  }

  torch::Tensor forward(const torch::Tensor& latents) {
    auto sample = conv_in_(latents);
    sample = mid_block_(sample);
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      sample = up_blocks_[i]->as<UpDecoderBlock2D>()->forward(sample);
    }
    sample = conv_norm_out_(sample);
    sample = conv_act_(sample);
    sample = conv_out_(sample);
    return sample;
  }

  void load_state_dict(const StateDict& state_dict) {
    LOG(INFO) << "Loading state_dict for VAEDecoder";
    // conv_in_
    state_dict.copy_tensor_to("conv_in.weight", conv_in_->weight);
    state_dict.copy_tensor_to("conv_in.bias", conv_in_->bias);
    // mid_block_
    //  mid_block_ is a UNetMidBlock2D, so we load its state
    mid_block_->load_state_dict(state_dict.get_dict_with_prefix("mid_block."));
    for (size_t i = 0; i < up_blocks_->size(); ++i) {
      up_blocks_[i]->as<UpDecoderBlock2D>()->load_state_dict(
          state_dict.get_dict_with_prefix("up_blocks." + std::to_string(i) +
                                          "."));
    }
    // conv_norm_out_
    state_dict.copy_tensor_to("conv_norm_out.weight", conv_norm_out_->weight);
    state_dict.copy_tensor_to("conv_norm_out.bias", conv_norm_out_->bias);
    // conv_out_
    state_dict.copy_tensor_to("conv_out.weight", conv_out_->weight);
    state_dict.copy_tensor_to("conv_out.bias", conv_out_->bias);
  }

 private:
  torch::nn::Conv2d conv_in_{nullptr};
  UNetMidBlock2D mid_block_{nullptr};
  torch::nn::ModuleList up_blocks_;
  torch::nn::GroupNorm conv_norm_out_{nullptr};
  torch::nn::Functional conv_act_{nullptr};
  torch::nn::Conv2d conv_out_{nullptr};
};
TORCH_MODULE(VAEDecoder);

// VAE implementation, including encoder and decoder
class VAEImpl : public torch::nn::Module {
 public:
  VAEImpl(const ModelContext& context) : args_(context.get_model_args()) {
    encoder_ = register_module("encoder", VAEEncoder(context));
    decoder_ = register_module("decoder", VAEDecoder(context));
    if (args_.use_quant_conv()) {
      quant_conv_ = register_module(
          "quant_conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(
              2 * args_.latent_channels(), 2 * args_.latent_channels(), 1)));
    }
    if (args_.use_post_quant_conv()) {
      post_quant_conv_ = register_module(
          "post_quant_conv",
          torch::nn::Conv2d(torch::nn::Conv2dOptions(
              args_.latent_channels(), args_.latent_channels(), 1)));
    }

    auto dtype = context.get_tensor_options().dtype().toScalarType();
    encoder_->to(dtype);
    decoder_->to(dtype);
    if (args_.use_quant_conv()) {
      quant_conv_->to(dtype);
    }
    if (args_.use_post_quant_conv()) {
      post_quant_conv_->to(dtype);
    }
  }

  torch::Tensor encode(const torch::Tensor& images) {
    auto enc = encoder_(images);
    if (args_.use_quant_conv()) {
      enc = quant_conv_(enc);
    }
    return enc;
  }

  torch::Tensor decode(const torch::Tensor& latents) {
    torch::Tensor processed_latents = latents;

    if (args_.use_post_quant_conv()) {
      processed_latents = post_quant_conv_(processed_latents);
    }

    auto dec = decoder_(processed_latents);
    return dec;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      encoder_->load_state_dict(state_dict->get_dict_with_prefix("encoder."));
      decoder_->load_state_dict(state_dict->get_dict_with_prefix("decoder."));
      if (args_.use_quant_conv()) {
        state_dict->copy_tensor_to("quant_conv.weight", quant_conv_->weight);
        state_dict->copy_tensor_to("quant_conv.bias", quant_conv_->bias);
      }
      if (args_.use_post_quant_conv()) {
        state_dict->copy_tensor_to("post_quant_conv.weight",
                                   post_quant_conv_->weight);
        state_dict->copy_tensor_to("post_quant_conv.bias",
                                   post_quant_conv_->bias);
      }
    }
    LOG(INFO) << "VAE model loaded successfully.";
  }

 private:
  VAEEncoder encoder_{nullptr};
  VAEDecoder decoder_{nullptr};
  torch::nn::Conv2d quant_conv_{nullptr};
  torch::nn::Conv2d post_quant_conv_{nullptr};
  bool use_post_quant_conv_{false};
  ModelArgs args_;
};
TORCH_MODULE(VAE);

// register the VAE model with the model registry
REGISTER_MODEL_ARGS(AutoencoderKL, [&] {
  LOAD_ARG_OR(in_channels, "in_channels", 3);
  LOAD_ARG_OR(out_channels, "out_channels", 3);
  LOAD_ARG_OR(down_block_types,
              "down_block_types",
              (std::vector<std::string>{"DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D",
                                        "DownEncoderBlock2D"}));
  LOAD_ARG_OR(up_block_types,
              "up_block_types",
              (std::vector<std::string>{"UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D",
                                        "UpDecoderBlock2D"}));
  LOAD_ARG_OR(block_out_channels,
              "block_out_channels",
              (std::vector<int64_t>{128, 256, 512, 512}));
  LOAD_ARG_OR(layers_per_block, "layers_per_block", 2);
  LOAD_ARG_OR(latent_channels, "latent_channels", 16);
  LOAD_ARG_OR(norm_num_groups, "norm_num_groups", 32);
  LOAD_ARG_OR(sample_size, "sample_size", 1024);
  LOAD_ARG_OR(scale_factor, "scale_factor", 0.3611f);
  LOAD_ARG_OR(shift_factor, "shift_factor", 0.1159f);
  LOAD_ARG_OR(mid_block_add_attention, "mid_block_add_attention", true);
  LOAD_ARG_OR(force_upcast, "force_upcast", true);
  LOAD_ARG_OR(use_quant_conv, "use_quant_conv", false);
  LOAD_ARG_OR(use_post_quant_conv, "use_post_quant_conv", false);
});
}  // namespace xllm