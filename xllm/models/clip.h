#pragma once
// 标准库
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// PyTorch 核心
#include <c10/core/ScalarType.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/torch.h>

// 项目内部头文件
#include <glog/logging.h>  // 需包含glog头文件

#include "chat_template/common_chat_template.h"
#include "common/tensor_helper.h"
#include "kv_cache/kv_cache.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/layernorm.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/qkv_linear.h"
#include "model_loader/state_dict.h"
#include "models/base/model_args.h"
#include "models/base/model_registry.h"
#include "models/base/parameters.h"
#include "models/llama.h"
#include "processors/clip_image_processor.h"

namespace llm::hf {
// Clip model ref to:
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L152

torch::Tensor quick_gelu(torch::Tensor x) {
  return x * torch::sigmoid(1.702f * x).to(torch::kFloat32);
}

// 生成4D因果注意力掩码 (batch_size, 1, seq_len, seq_len)
torch::Tensor _create_4d_causal_attention_mask(
    torch::IntArrayRef
        input_shape,  // 期望输入形状为 [batch_size, seq_len, ...]
    torch::Dtype dtype,
    torch::Device device) {
  const int64_t batch_size = input_shape[0];
  const int64_t seq_len = input_shape[1];

  // 创建二维因果掩码 (seq_len, seq_len)
  auto mask = torch::full({seq_len, seq_len},
                          -std::numeric_limits<float>::infinity(),
                          dtype)
                  .triu(1);  // 保留上三角部分（不含对角线）

  // 扩展为4D张量 (batch_size, 1, seq_len, seq_len)
  return mask
      .unsqueeze(0)  // 添加batch维度
      .unsqueeze(0)  // 添加头维度
      .expand({batch_size, 1, seq_len, seq_len})
      .to(device);  // 确保设备一致
}

std::vector<char> get_the_bytes(std::string filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}

// class CLIPVisionEmbeddingImpl : public torch::nn::Module {
//  public:
//   CLIPVisionEmbeddingImpl(const ModelArgs& args,
//                           const QuantArgs& quant_args,
//                           const ParallelArgs& parallel_args,
//                           const torch::TensorOptions& options) {
//     // auto options_float32 =
//     options.dtype(torch::kFloat32).device(torch::kCPU); embed_dim_ =
//     args.hidden_size(); image_size_ = args.mm_image_size(); class_embedding_
//     = register_parameter("class_embedding",
//                                           torch::randn({embed_dim_},
//                                           options));
//     patch_embedding_ = register_module(
//         "patch_embedding",
//         torch::nn::Conv2d(torch::nn::Conv2dOptions(args.num_channels(),
//                                                    embed_dim_,
//                                                    args.mm_patch_size())
//                               .stride(args.mm_patch_size())
//                               .bias(false)));
//     patch_embedding_->weight.set_data(patch_embedding_->weight.to(options));

//     auto num_patches = (args.mm_image_size() / args.mm_patch_size()) *
//                        (args.mm_image_size() / args.mm_patch_size());
//     auto num_positions = num_patches + 1;
//     position_embedding_ =
//         register_parameter("position_embedding",
//                            torch::randn({num_positions, embed_dim_},
//                            options));
//     position_ids_ = register_buffer(
//         "position_ids",
//         torch::arange(0, num_positions, torch::kLong).unsqueeze(0));
//   }

//   torch::Tensor forward(const torch::Tensor& pixel_values) {
//     int64_t batch_size = pixel_values.size(0);
//     auto patch_embeds =
//         patch_embedding_->forward(pixel_values).flatten(2).transpose(1, 2);
//     auto class_embeds = class_embedding_.expand({batch_size, 1, embed_dim_});
//     auto embeddings = torch::cat({class_embeds, patch_embeds}, 1);
//     embeddings += position_embedding_.index({position_ids_});
//     return embeddings;
//   }

//   // load the weight from the checkpoint
//   void load_state_dict(const StateDict& state_dict) {
//     const auto cls = state_dict.get_tensor("class_embedding");
//     if (cls.defined()) {
//       DCHECK_EQ(cls.sizes(), class_embedding_.sizes())
//           << "class_embedding size mismatch for " << name();
//       class_embedding_.data().copy_(cls);
//       is_class_embedding_loaded = true;
//     }

//     const auto pos = state_dict.get_tensor("position_embedding.weight");
//     if (pos.defined()) {
//       CHECK_EQ(pos.sizes(), position_embedding_.sizes())
//           << "position_embedding weight size mismatch for " << name();
//       position_embedding_.data().copy_(pos);
//       is_position_embedding_loaded = true;
//     }

//     const auto weight = state_dict.get_tensor("patch_embedding.weight");
//     if (weight.defined()) {
//       DCHECK_EQ(patch_embedding_->weight.sizes(), weight.sizes())
//           << "patch_embedding weight size mismatch for " << name();
//       patch_embedding_->weight.data().copy_(weight);
//       is_patch_embedding_loaded = true;
//     }
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     CHECK(is_class_embedding_loaded)
//         << "weight is not loaded for " << prefix + "class_embedding";
//     CHECK(is_position_embedding_loaded)
//         << "weight is not loaded for " << prefix +
//         "position_embedding.weight";
//     CHECK(is_patch_embedding_loaded)
//         << "weight is not loaded for " << prefix + "patch_embedding.weight";
//   }

//  private:
//   int64_t embed_dim_;
//   int64_t image_size_;
//   bool is_class_embedding_loaded{false};
//   bool is_position_embedding_loaded{false};
//   bool is_patch_embedding_loaded{false};

//   torch::Tensor class_embedding_;
//   torch::Tensor position_ids_;
//   torch::nn::Conv2d patch_embedding_{nullptr};
//   torch::Tensor position_embedding_{nullptr};
// };
// TORCH_MODULE(CLIPVisionEmbedding);

class CLIPTextEmbeddingImpl : public torch::nn::Module {
 public:
  CLIPTextEmbeddingImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options) {
    // auto options_float32 =
    // options.dtype(torch::kFloat32).device(torch::kCPU);
    LOG(INFO) << "register CLIPTextEmbedding with options: " << args;
    token_embedding_ =
        register_module("token_embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(
                            args.vocab_size(), args.hidden_size())));
    token_embedding_->weight.set_data(token_embedding_->weight.to(options));
    LOG(INFO) << "CLIPTextEmbedding token_embedding registered.";
    position_embedding_ = register_parameter(
        "position_embedding",
        torch::randn({args.max_position_embeddings(), args.hidden_size()},
                     options));
    LOG(INFO) << "CLIPTextEmbedding position_embedding registered.";
    position_ids_ = register_buffer(
        "position_ids",
        torch::arange(0, args.max_position_embeddings(), torch::kLong)
            .unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    int64_t batch_size = input_ids.size(0);
    int64_t seq_length = input_ids.size(1);
    int64_t max_position_embedding = position_embedding_.size(0);
    if (seq_length > max_position_embedding) {
      throw std::runtime_error(
          "Sequence length exceeds max_position_embeddings: " +
          std::to_string(seq_length) + " > " +
          std::to_string(max_position_embedding));
    }

    // token embeddings [B, S, H]
    torch::Tensor inputs_embeds = token_embedding_->forward(input_ids);

    torch::Tensor position_ids = position_ids_.index(
        {torch::indexing::Slice(), torch::indexing::Slice(None, seq_length)});
    torch::Tensor embeddings =
        inputs_embeds + position_embedding_.index({position_ids});
    return embeddings;
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto tok_emb = state_dict.get_tensor("token_embedding.weight");
    if (tok_emb.defined()) {
      DCHECK_EQ(token_embedding_->weight.sizes(), tok_emb.sizes())
          << "patch_embedding weight size mismatch for " << name();
      token_embedding_->weight.data().copy_(tok_emb);
      is_token_embedding_loaded = true;
    }

    const auto pos = state_dict.get_tensor("position_embedding.weight");
    if (pos.defined()) {
      CHECK_EQ(pos.sizes(), position_embedding_.sizes())
          << "position_embedding weight size mismatch for " << name();
      position_embedding_.data().copy_(pos);
      is_position_embedding_loaded = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_position_embedding_loaded)
        << "weight is not loaded for " << prefix + "position_embedding.weight";
    CHECK(is_token_embedding_loaded)
        << "weight is not loaded for " << prefix + "token_embedding.weight";
  }

 private:
  bool is_position_embedding_loaded{false};
  bool is_token_embedding_loaded{false};

  torch::Tensor position_ids_;
  torch::nn::Embedding token_embedding_{nullptr};
  torch::Tensor position_embedding_{nullptr};
};
TORCH_MODULE(CLIPTextEmbedding);

class CLIPMLPImpl : public torch::nn::Module {
 public:
  CLIPMLPImpl(const ModelArgs& args,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options) {
    // https://github.com/huggingface/transformers/.../configuration_clip.py
    // auto options_float32 =
    // options.dtype(torch::kFloat32).device(torch::kCPU);
    act_ = quick_gelu;
    CHECK(act_ != nullptr);

    fc1_ = register_module(
        "fc1",
        torch::nn::Linear(torch::nn::LinearOptions(args.hidden_size(),
                                                   args.intermediate_size())
                              .bias(true)));
    fc2_ = register_module(
        "fc2",
        torch::nn::Linear(torch::nn::LinearOptions(args.intermediate_size(),
                                                   args.hidden_size())
                              .bias(true)));

    // fc1_ = register_module("fc1",
    //                        ColumnParallelLinear(args.hidden_size(),
    //                                             args.intermediate_size(),
    //                                             /*bias=*/true,
    //                                             /*gather_output=*/false,
    //                                             quant_args,
    //                                             parallel_args,
    //                                             options_float32));
    // fc2_ = register_module("fc2",
    //                        RowParallelLinear(args.intermediate_size(),
    //                                          args.hidden_size(),
    //                                          /*bias=*/true,
    //                                          /*input_is_parallelized*/ true,
    //                                          quant_args,
    //                                          parallel_args,
    //                                          options_float32));

    // Handle FC layers
    fc1_->weight.set_data(fc1_->weight.to(options));
    fc2_->weight.set_data(fc2_->weight.to(options));

    // Handle FC bias terms
    fc1_->bias.set_data(fc1_->bias.to(options));
    fc2_->bias.set_data(fc2_->bias.to(options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return fc2_(act_(fc1_(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto fc1_weight = state_dict.get_tensor("fc1.weight");
    if (fc1_weight.defined()) {
      DCHECK_EQ(fc1_weight.sizes(), fc1_->weight.sizes())
          << "fc1 weight size mismatch";
      fc1_->weight.data().copy_(fc1_weight);
    }

    const auto fc1_bias = state_dict.get_tensor("fc1.bias");
    if (fc1_bias.defined()) {
      DCHECK_EQ(fc1_bias.sizes(), fc1_->bias.sizes())
          << "fc1 bias size mismatch";
      fc1_->bias.data().copy_(fc1_bias);
    }

    const auto fc2_weight = state_dict.get_tensor("fc2.weight");
    if (fc2_weight.defined()) {
      DCHECK_EQ(fc2_weight.sizes(), fc2_->weight.sizes())
          << "fc2 weight size mismatch";
      fc2_->weight.data().copy_(fc2_weight);
    }

    const auto fc2_bias = state_dict.get_tensor("fc2.bias");
    if (fc2_bias.defined()) {
      DCHECK_EQ(fc2_bias.sizes(), fc2_->bias.sizes())
          << "fc2 bias size mismatch";
      fc2_->bias.data().copy_(fc2_bias);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    // fc1_->verify_loaded_weights(prefix + "fc1.");
    // fc2_->verify_loaded_weights(prefix + "fc2.");
    // CHECK(isload) << "weight is not loaded for " << prefix + "fc.weight";
  }

 private:
  ActFunc act_{nullptr};
  torch::nn::Linear fc1_{nullptr};
  torch::nn::Linear fc2_{nullptr};
  // bool isload{false};
};
TORCH_MODULE(CLIPMLP);

// TODO: Optimize CLIPAttention
class CLIPAttentionImpl : public torch::nn::Module {
 public:
  CLIPAttentionImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options) {
    // auto options_float32 =
    // options.dtype(torch::kFloat32).device(torch::kCPU);
    CHECK(args.hidden_size() % args.num_attention_heads() == 0);

    head_dim_ = args.head_dim();
    embed_dim_ = args.hidden_size();
    num_heads_ = args.num_attention_heads();
    const int64_t n_local_heads = num_heads_;

    qkv_sizes_ = {n_local_heads * args.head_dim(),
                  n_local_heads * args.head_dim(),
                  n_local_heads * args.head_dim()};

    scale_ = 1.0f / std::sqrt(static_cast<float>(args.head_dim()));
    dropout_ = args.attention_dropout();
    LOG(INFO) << "CLIPAttentionImpl with options: " << options
              << ", head_dim: " << head_dim_ << ", embed_dim: " << embed_dim_
              << ", num_heads: " << num_heads_;
    q_proj_ = register_module(
        "q_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(args.hidden_size(), num_heads_ * head_dim_)
                .bias(true)));
    k_proj_ = register_module(
        "k_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(args.hidden_size(), num_heads_ * head_dim_)
                .bias(true)));
    v_proj_ = register_module(
        "v_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(args.hidden_size(), num_heads_ * head_dim_)
                .bias(true)));
    o_proj_ = register_module(
        "o_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(args.hidden_size(), args.hidden_size())
                .bias(true)));
    LOG(INFO) << "CLIPAttentionImpl modules registered.";
    q_proj_->weight.set_data(q_proj_->weight.to(options));
    k_proj_->weight.set_data(k_proj_->weight.to(options));
    v_proj_->weight.set_data(v_proj_->weight.to(options));
    o_proj_->weight.set_data(o_proj_->weight.to(options));

    // Handle bias terms
    q_proj_->bias.set_data(q_proj_->bias.to(options));
    k_proj_->bias.set_data(k_proj_->bias.to(options));
    v_proj_->bias.set_data(v_proj_->bias.to(options));
    o_proj_->bias.set_data(o_proj_->bias.to(options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto bsz = hidden_states.size(0);
    auto tgt_len = hidden_states.size(1);

    // 使用 RowParallelLinear 分别生成 query, key, value
    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // 调整形状为 [batch_size, num_heads, seq_len, head_dim]
    query_states = shape(query_states, tgt_len, bsz);
    key_states = shape(key_states, -1, bsz);
    value_states = shape(value_states, -1, bsz);

    auto src_len = key_states.size(1);
    auto attn_weights =
        torch::matmul(query_states, key_states.transpose(-1, -2)) * scale_;

    auto causal_mask =
        torch::full({tgt_len, tgt_len}, -std::numeric_limits<float>::infinity())
            .to(attn_weights.device());
    causal_mask.triu_(1);
    // causal_mask = causal_mask.to(attn_weights.device());

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0);
    causal_mask = causal_mask.expand({bsz, 1, tgt_len, tgt_len});

    attn_weights = attn_weights + causal_mask;
    attn_weights = torch::softmax(attn_weights, -1, torch::kFloat32);
    auto attn_probs = torch::dropout(attn_weights, dropout_, false);
    auto attn_output = torch::matmul(attn_probs, value_states);

    DCHECK_EQ(attn_output.sizes(),
              torch::IntArrayRef({bsz * num_heads_, tgt_len, head_dim_}));
    attn_output =
        attn_output
            .view(torch::IntArrayRef({bsz, num_heads_, tgt_len, head_dim_}))
            .transpose(1, 2)
            .contiguous();
    attn_output =
        attn_output.view(torch::IntArrayRef({bsz, tgt_len, embed_dim_}));

    return o_proj_(attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto q_proj_weight = state_dict.get_tensor("q_proj.weight");
    if (q_proj_weight.defined()) {
      DCHECK_EQ(q_proj_weight.sizes(), q_proj_->weight.sizes())
          << "q_proj weight size mismatch";
      q_proj_->weight.data().copy_(q_proj_weight);
    }
    const auto q_proj_bias = state_dict.get_tensor("q_proj.bias");
    if (q_proj_bias.defined()) {
      DCHECK_EQ(q_proj_bias.sizes(), q_proj_->bias.sizes())
          << "q_proj bias size mismatch";
      q_proj_->bias.data().copy_(q_proj_bias);
    }

    const auto k_proj_weight = state_dict.get_tensor("k_proj.weight");
    if (k_proj_weight.defined()) {
      DCHECK_EQ(k_proj_weight.sizes(), k_proj_->weight.sizes())
          << "k_proj weight size mismatch";
      k_proj_->weight.data().copy_(k_proj_weight);
    }
    const auto k_proj_bias = state_dict.get_tensor("k_proj.bias");
    if (k_proj_bias.defined()) {
      DCHECK_EQ(k_proj_bias.sizes(), k_proj_->bias.sizes())
          << "k_proj bias size mismatch";
      k_proj_->bias.data().copy_(k_proj_bias);
    }

    const auto v_proj_weight = state_dict.get_tensor("v_proj.weight");
    if (v_proj_weight.defined()) {
      DCHECK_EQ(v_proj_weight.sizes(), v_proj_->weight.sizes())
          << "v_proj weight size mismatch";
      v_proj_->weight.data().copy_(v_proj_weight);
    }
    const auto v_proj_bias = state_dict.get_tensor("v_proj.bias");
    if (v_proj_bias.defined()) {
      DCHECK_EQ(v_proj_bias.sizes(), v_proj_->bias.sizes())
          << "v_proj bias size mismatch";
      v_proj_->bias.data().copy_(v_proj_bias);
    }

    const auto o_proj_weight = state_dict.get_tensor("out_proj.weight");
    if (o_proj_weight.defined()) {
      DCHECK_EQ(o_proj_weight.sizes(), o_proj_->weight.sizes())
          << "o_proj weight size mismatch";
      o_proj_->weight.data().copy_(o_proj_weight);
    }
    const auto o_proj_bias = state_dict.get_tensor("out_proj.bias");
    if (o_proj_bias.defined()) {
      DCHECK_EQ(o_proj_bias.sizes(), o_proj_->bias.sizes())
          << "o_proj bias size mismatch";
      o_proj_->bias.data().copy_(o_proj_bias);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    // qkv_proj_->verify_loaded_weights(prefix + "[q_proj,k_proj,v_proj].");
    // q_proj_->verify_loaded_weights(prefix + "q_proj.");
    // k_proj_->verify_loaded_weights(prefix + "k_proj.");
    // v_proj_->verify_loaded_weights(prefix + "v_proj.");
    // o_proj_->verify_loaded_weights(prefix + "out_proj.");
    // CHECK(isload) << "weight is not loaded for " << prefix + "proj.weight";
  }

 private:
  torch::Tensor shape(torch::Tensor tensor, int64_t seq_len, int64_t bsz) {
    return tensor.view({bsz, seq_len, num_heads_, head_dim_})
        .transpose(1, 2)
        .contiguous();
  }

 private:
  int64_t embed_dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  float scale_;
  float dropout_;
  std::vector<int64_t> qkv_sizes_;

  torch::nn::Linear o_proj_{nullptr};
  torch::nn::Linear q_proj_{nullptr};
  torch::nn::Linear k_proj_{nullptr};
  torch::nn::Linear v_proj_{nullptr};
  Attention atten_{nullptr};
};
TORCH_MODULE(CLIPAttention);

class CLIPEncoderLayerImpl : public torch::nn::Module {
 public:
  CLIPEncoderLayerImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options) {
    // auto option_float32 = options.dtype(torch::kFloat32).device(torch::kCPU);
    LOG(INFO) << "register CLIPEncoderLayer with options: " << options;
    self_attn_ = register_module(
        "self_attn", CLIPAttention(args, quant_args, parallel_args, options));
    LOG(INFO) << "CLIPEncoderLayer self_attn registered.";
    layer_norm1_ = register_module(
        "layer_norm1",
        LayerNormInst(
            // args.mm_hidden_size(), args.layer_norm_eps(), true, options));
            args.hidden_size(),
            args.layer_norm_eps(),
            true,
            options));
    LOG(INFO) << "CLIPEncoderLayer layer_norm1 registered.";
    layer_norm2_ = register_module(
        "layer_norm2",
        LayerNormInst(
            // args.mm_hidden_size(), args.layer_norm_eps(), true, options));
            args.hidden_size(),
            args.layer_norm_eps(),
            true,
            options));
    LOG(INFO) << "CLIPEncoderLayer layer_norm2 registered.";

    mlp_ = register_module("mlp",
                           CLIPMLP(args, quant_args, parallel_args, options));
  }

  // TODO: self_attn, attention_mask, causal_attention_mask
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto residual = hidden_states;
    const auto& layer_norm1 = layer_norm1_(hidden_states);
    auto h = self_attn_(layer_norm1) + residual;
    residual = h;
    h = layer_norm2_(h);
    h = mlp_(h);
    h += residual;
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    LOG(INFO) << "Loading state dict for CLIPEncoderLayer.";
    self_attn_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
    LOG(INFO) << "CLIPEncoderLayer self_attn loaded.";
    layer_norm1_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm1."));
    LOG(INFO) << "CLIPEncoderLayer layer_norm1 loaded.";
    for (auto& param : layer_norm1_->parameters()) {
      param = param.to(torch::kFloat32);  // Cast to float32
    }
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
    LOG(INFO) << "CLIPEncoderLayer mlp loaded.";
    layer_norm2_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm2."));
    LOG(INFO) << "CLIPEncoderLayer layer_norm2 loaded.";
    for (auto& param : layer_norm1_->parameters()) {
      param = param.to(torch::kFloat32);  // Cast to float32
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    layer_norm1_->verify_loaded_weights(prefix + "layer_norm1.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    layer_norm2_->verify_loaded_weights(prefix + "layer_norm2.");
  }

 private:
  CLIPAttention self_attn_{nullptr};
  LayerNormInst layer_norm1_{nullptr};
  CLIPMLP mlp_{nullptr};
  LayerNormInst layer_norm2_{nullptr};
};
TORCH_MODULE(CLIPEncoderLayer);

// encoder只返回最后一层的hidden_state
class CLIPEncoderImpl : public torch::nn::Module {
 public:
  CLIPEncoderImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options) {
    // auto options = options_.dtype(torch::kFloat32);
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.num_hidden_layers());
    for (int32_t i = 0; i < args.num_hidden_layers(); i++) {
      auto block = CLIPEncoderLayer(args, quant_args, parallel_args, options);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // Output hidden states for last intermediate layers
  torch::Tensor forward(const torch::Tensor& embeddings) {
    bool output_hidden_states = false;
    bool output_attentions = false;
    c10::optional<torch::Tensor> attention_mask = c10::nullopt;
    c10::optional<torch::Tensor> head_mask = c10::nullopt;
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<torch::Tensor> all_attentions;
    std::vector<torch::Tensor> encoder_states;

    // all_attention没有保留
    auto hidden_states = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
      encoder_states.emplace_back(hidden_states);
      auto& layer = layers_[i];
      hidden_states = layer(hidden_states);
    }
    if (output_hidden_states) encoder_states.emplace_back(hidden_states);

    std::vector<torch::Tensor> outputs = {hidden_states};
    if (output_hidden_states) {
      // todo
    }
    if (output_attentions) {
      // todo
    }
    return outputs[0];
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<CLIPEncoderLayer> layers_;
};
TORCH_MODULE(CLIPEncoder);

// class CLIPVisionTransformerImpl : public torch::nn::Module {
//  public:
//   CLIPVisionTransformerImpl(const ModelArgs& args,
//                             const QuantArgs& quant_args,
//                             const ParallelArgs& parallel_args,
//                             const torch::TensorOptions& options) {
//     embeddings_ = register_module(
//         "embeddings",
//         CLIPVisionEmbedding(args, quant_args, parallel_args, options));
//     pre_layernorm_ = register_module(
//         "pre_layernorm",
//         LayerNormInst(
//             args.mm_hidden_size(), args.mm_layer_norm_eps(), true, options));

//     encoder_ = register_module(
//         "encoder", CLIPEncoder(args, quant_args, parallel_args, options));
//     post_layernorm_ = register_module(
//         "post_layernorm",
//         LayerNormInst(
//             args.mm_hidden_size(), args.mm_layer_norm_eps(), true, options));
//   }

//   // std::vector<torch::Tensor> forward(const torch::Tensor& pixel_values) {
//   torch::Tensor forward(const torch::Tensor& pixel_values) {
//     auto hidden_states = embeddings_->forward(pixel_values);
//     hidden_states = pre_layernorm_->forward(hidden_states);

//     auto pool_output = encoder_->forward(hidden_states);
//     auto encoder_output = post_layernorm_->forward(pool_output);
//     return encoder_output;
//   }

//   // load the weight from the checkpoint
//   void load_state_dict(const StateDict& state_dict) {
//     embeddings_->load_state_dict(
//         state_dict.get_dict_with_prefix("embeddings."));
//     pre_layernorm_->load_state_dict(
//         state_dict.get_dict_with_prefix("pre_layrnorm."));
//     encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
//     post_layernorm_->load_state_dict(
//         state_dict.get_dict_with_prefix("post_layernorm."));
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     embeddings_->verify_loaded_weights(prefix + "embeddings.");
//     pre_layernorm_->verify_loaded_weights(prefix + "pre_layrnorm.");
//     encoder_->verify_loaded_weights(prefix + "encoder.");
//     post_layernorm_->verify_loaded_weights(prefix + "post_layernorm.");
//   }

//  private:
//   CLIPVisionEmbedding embeddings_{nullptr};
//   LayerNormInst pre_layernorm_{nullptr};
//   CLIPEncoder encoder_{nullptr};
//   LayerNormInst post_layernorm_{nullptr};
// };
// TORCH_MODULE(CLIPVisionTransformer);

// // Follow implementation: https://github.com/huggingface/transformers
// class CLIPVisionModelImpl : public torch::nn::Module {
//  public:
//   CLIPVisionModelImpl(const ModelArgs& args,
//                       const QuantArgs& quant_args,
//                       const ParallelArgs& parallel_args,
//                       const torch::TensorOptions& options) {
//     transformer_ = register_module(
//         "transformer",
//         CLIPVisionTransformer(args, quant_args, parallel_args, options));
//   }

//   // return hidden_state (TODO support return: output_attention, return_dict)
//   // std::vector<torch::Tensor> forward(const torch::Tensor& images) {
//   torch::Tensor forward(const torch::Tensor& images,
//                         const torch::Tensor& positions,
//                         std::vector<llm::KVCache>& kv_caches,
//                         const llm::InputParameters& parameters) {
//     return transformer_->forward(images);
//   }

//   // load the weight from the checkpoint
//   void load_state_dict(const StateDict& state_dict) {
//     transformer_->load_state_dict(
//         state_dict.get_dict_with_prefix("vision_model."));
//   }

//   void verify_loaded_weights(const std::string& prefix) const {
//     transformer_->verify_loaded_weights(prefix + "vision_model.");
//   }
//   llm::hf::LlmHead get_lm_head() {
//     // 返回一个空的 LlmHead 对象
//     LOG(INFO) << "VAE does not support LLM head. Returning an empty
//     LlmHead."; static const llm::hf::LlmHead empty_head{
//         nullptr};       // 静态对象仅初始化一次，规避多次构造
//     return empty_head;  // 返回拷贝（若拷贝构造合法）
//   }
//   llm::hf::AtbWordEmbedding get_word_embedding() {
//     LOG(INFO) << "VAE does not support word embeddings. Returning an empty "
//                  "AtbWordEmbedding.";
//     static const llm::hf::AtbWordEmbedding empty_embedding{nullptr};
//     return empty_embedding;
//   }
//   // 空实现：set_word_embedding（VAE不需要词嵌入）
//   void set_word_embedding(llm::hf::AtbWordEmbedding& embedding) {
//     LOG(INFO) << "VAE does not support setting word embedding. This method is
//     "
//                  "a no-op.";
//     // 空实现
//   }
//   // 空实现：logits（VAE不需要语言模型的logits）
//   torch::Tensor logits(const at::Tensor& hidden_states,
//                        const at::Tensor& seleted_idxes) {
//     LOG(INFO) << "VAE does not support logits. Returning an empty tensor.";
//     // 可选：返回空张量或抛出未实现异常
//     return torch::Tensor();
//   }
//   // 空实现：set_lm_head（VAE不需要设置语言模型头）
//   void set_lm_head(llm::hf::LlmHead& head) {
//     // 空实现（不做任何操作）
//     LOG(INFO)
//         << "VAE does not support setting LLM head. This method is a no-op.";
//   }
//   void load_model(std::unique_ptr<ModelLoader> loader) {
//     // VAE模型加载逻辑（如果有）
//     // LOG(INFO) << "VAE model loaded successfully.";

//     for (const auto& state_dict : *loader) {
//       transformer_->load_state_dict(
//           state_dict.get_dict_with_prefix("vision_model."));
//     }

//     // verify
//     transformer_->verify_loaded_weights("vision_model.");
//     LOG(INFO) << "clip_text_model loaded successfully.";
//   }

//  private:
//   CLIPVisionTransformer transformer_{nullptr};
// };
// TORCH_MODULE(CLIPVisionModel);

class CLIPTextTransformerImpl : public torch::nn::Module {
 public:
  CLIPTextTransformerImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options) {
    auto option_float32 =
        options.dtype(torch::kFloat32).device(options.device());
    embeddings_ = register_module(
        "embeddings",
        CLIPTextEmbedding(args, quant_args, parallel_args, option_float32));
    final_layer_norm_ = register_module(
        "final_layer_norm",
        LayerNormInst(
            args.hidden_size(), args.layer_norm_eps(), true, option_float32));
    encoder_ = register_module(
        "encoder",
        CLIPEncoder(args, quant_args, parallel_args, option_float32));
    eos_token_id = args.eos_token_id();
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    if (!input_ids.defined()) {
      throw std::runtime_error("You have to specify input_ids");
    }
    auto input_shape = input_ids.sizes();
    auto reshaped_input_ids = input_ids.view({-1, input_shape.back()});
    auto hidden_states = embeddings_->forward(reshaped_input_ids);
    auto encoder_output = encoder_->forward(hidden_states);
    auto last_hidden_state = final_layer_norm_->forward(encoder_output);
    return last_hidden_state;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(
        state_dict.get_dict_with_prefix("embeddings."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    final_layer_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("final_layer_norm."));

    for (auto& param : final_layer_norm_->parameters()) {
      param = param.to(torch::kFloat32);  // Cast to float32
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
    final_layer_norm_->verify_loaded_weights(prefix + "final_layer_norm.");
  }

 private:
  int64_t eos_token_id;
  CLIPTextEmbedding embeddings_{nullptr};
  CLIPEncoder encoder_{nullptr};
  LayerNormInst final_layer_norm_{nullptr};
};
TORCH_MODULE(CLIPTextTransformer);

class CLIPTextModelImpl : public torch::nn::Module {
 public:
  CLIPTextModelImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options,
                    torch::Device device = torch::kCPU,
                    torch::ScalarType dtype = torch::kFloat32) {
    device_ = device;
    dtype_ = dtype;
    auto options_float32 = options.dtype(dtype_).device(device_);
    eos_token_id = args.eos_token_id();
    transformer_ = register_module(
        "transformer",
        CLIPTextTransformer(args, quant_args, parallel_args, options_float32));
  }

  torch::Tensor forward(std::vector<int64_t> input_ids_data) {
    torch::Tensor input_ids =
        torch::tensor(input_ids_data, torch::kLong).view({1, -1});
    input_ids = input_ids.to(device_);
    auto last_hidden_states = transformer_->forward(input_ids);
    int64_t batch_size = last_hidden_states.size(0);
    auto device = last_hidden_states.device();
    torch::Tensor batch_indices = torch::arange(batch_size, device);
    torch::Tensor end_pos;
    if (eos_token_id == 2) {
      auto argmax_result = input_ids.to(device).max(1);
      end_pos = std::get<1>(argmax_result);
    } else {
      torch::Tensor eos_mask =
          (input_ids == eos_token_id).to(device, torch::kInt);
      auto argmax_result = eos_mask.max(1);
      end_pos = std::get<1>(argmax_result);
    }
    torch::Tensor pooled_output =
        last_hidden_states.index({batch_indices, end_pos});
    return pooled_output;
  }

  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(
        state_dict.get_dict_with_prefix("text_model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    transformer_->verify_loaded_weights(prefix + ".");
  }
  void load_model(std::unique_ptr<ModelLoader> loader) {
    LOG(INFO) << "Loading CLIPTextModel from ModelLoader...";
    for (const auto& state_dict : *loader) {
      transformer_->load_state_dict(
          state_dict.get_dict_with_prefix("text_model."));
    }

    // verify
    transformer_->verify_loaded_weights("text_model.");
    LOG(INFO) << "clip_text_model loaded successfully.";
  }

 private:
  int64_t eos_token_id;
  CLIPTextTransformer transformer_{nullptr};
  torch::Device device_{torch::kCPU};  // Default to CPU, can be set later
  torch::ScalarType dtype_{torch::kFloat32};
};
TORCH_MODULE(CLIPTextModel);

REGISTER_MODEL_ARGS(clip_text_model, [&] {
  LOAD_ARG_OR(model_type, "model_type", "clip_text_model");

  LOAD_ARG_OR(vocab_size, "vocab_size", 49408);
  LOAD_ARG_OR(hidden_size, "hidden_size", 768);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 3072);
  LOAD_ARG_OR(projection_dim, "projection_dim", 768);
  LOAD_ARG_OR(num_hidden_layers, "num_hidden_layers", 12);
  LOAD_ARG_OR(num_attention_heads, "num_attention_heads", 12);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 77);
  LOAD_ARG_OR(hidden_act, "hidden_act", "quick_gelu");
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_eps", 1e-5f);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(initializer_factor, "initializer_factor", 1.0f);
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 1);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
});
}  // namespace llm::hf