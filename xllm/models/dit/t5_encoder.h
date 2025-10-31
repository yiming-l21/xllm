#pragma once
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
#include "xllm/core/kernels/npu/xllm_ops/add_matmul.h"

namespace xllm {
// T5 model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/tree/main/src/transformers/models/t5
class T5LayerNormImpl : public torch::nn::Module {
 public:
  torch::Tensor weight;
  double variance_epsilon;
  torch::Device device_;
  torch::ScalarType dtype_;

 public:
  T5LayerNormImpl(int64_t hidden_size,
                  double eps = 1e-6,
                  torch::Device device = torch::kCPU,
                  torch::ScalarType dtype = torch::kBFloat16)
      : variance_epsilon(eps), device_(device), dtype_(dtype) {
    weight = register_parameter(
        "weight", torch::ones({hidden_size}).to(device_).to(dtype_));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    auto variance = hidden_states.to(dtype_).pow(2).mean(-1, true);
    hidden_states = hidden_states * torch::rsqrt(variance + variance_epsilon);
    if (weight.dtype() == torch::kFloat16 ||
        weight.dtype() == torch::kBFloat16) {
      hidden_states = hidden_states.to(weight.dtype());
    }
    return weight * hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto weight_tensor = state_dict.get_tensor("weight");
    if (weight_tensor.defined()) {
      DCHECK_EQ(weight.sizes(), weight_tensor.sizes())
          << "weight size mismatch: expected " << weight.sizes() << " but got "
          << weight_tensor.sizes();
      weight.data().copy_(weight_tensor);
    }
  }
};
TORCH_MODULE(T5LayerNorm);

inline torch::Tensor gelu_new(const torch::Tensor& x) {
  // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
  const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
  return 0.5 * x *
         (1.0 +
          torch::tanh(sqrt_2_over_pi * (x + 0.044715 * torch::pow(x, 3))));
}

class T5DenseInterface : public torch::nn::Module {
 public:
  virtual torch::Tensor forward(const torch::Tensor& hidden_states) = 0;
  virtual void load_state_dict(const StateDict& state_dict) = 0;
};

class T5DenseActDenseImpl : public T5DenseInterface {
 public:
  T5DenseActDenseImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    wi_ = register_module(
        "wi",
        xllm_ops::DiTLinear(
            model_args.t5_d_model(), model_args.t5_d_ff(), false, options));
    wo_ = register_module(
        "wo",
        xllm_ops::DiTLinear(
            model_args.t5_d_ff(), model_args.t5_d_model(), false, options));

    // wi_->weight.set_data(wi_->weight.to(options));
    // wo_->weight.set_data(wo_->weight.to(options));
    if (model_args.t5_dense_act_fn() == "relu") {
      act_ = register_module("act", torch::nn::Functional(torch::relu));
    } else if (model_args.t5_dense_act_fn() == "gelu_new") {
      act_ = register_module("act", torch::nn::Functional(gelu_new));
    } else {
      throw std::invalid_argument("Unsupported activation function: " +
                                  model_args.t5_dense_act_fn());
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    // wi
    wi_->load_state_dict(state_dict.get_dict_with_prefix("wi."));
    // wo
    wo_->load_state_dict(state_dict.get_dict_with_prefix("wo."));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor hidden = wi_->forward(hidden_states);
    hidden = act_(hidden);
    hidden = wo_->forward(hidden);
    return hidden;
  }

 private:
  xllm_ops::DiTLinear wi_{nullptr};
  xllm_ops::DiTLinear wo_{nullptr};
  torch::nn::Functional act_{nullptr};
};

class T5DenseGatedActDenseImpl : public T5DenseInterface {
 public:
  T5DenseGatedActDenseImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    wi_0_ = register_module(
        "wi_0",
        xllm_ops::DiTLinear(
            model_args.t5_d_model(), model_args.t5_d_ff(), false, options));
    wi_1_ = register_module(
        "wi_1",
        xllm_ops::DiTLinear(
            model_args.t5_d_model(), model_args.t5_d_ff(), false, options));
    wo_ = register_module(
        "wo",
        xllm_ops::DiTLinear(
            model_args.t5_d_ff(), model_args.t5_d_model(), false, options));

    if (model_args.t5_dense_act_fn() == "relu") {
      act_ = register_module("act", torch::nn::Functional(torch::relu));
    } else if (model_args.t5_dense_act_fn() == "gelu_new") {
      act_ = register_module("act", torch::nn::Functional(gelu_new));
    } else {
      throw std::invalid_argument("Unsupported activation function: " +
                                  model_args.t5_dense_act_fn());
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    // wi_0
    wi_0_->load_state_dict(state_dict.get_dict_with_prefix("wi_0."));
    // wi_1
    wi_1_->load_state_dict(state_dict.get_dict_with_prefix("wi_1."));
    // wo
    wo_->load_state_dict(state_dict.get_dict_with_prefix("wo."));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor hidden_gelu = act_(wi_0_->forward(hidden_states));
    torch::Tensor hidden_linear = wi_1_->forward(hidden_states);
    torch::Tensor new_hidden_states = hidden_gelu * hidden_linear;
    new_hidden_states = wo_->forward(new_hidden_states);
    return new_hidden_states;
  }

 private:
  xllm_ops::DiTLinear wi_0_{nullptr};
  xllm_ops::DiTLinear wi_1_{nullptr};
  xllm_ops::DiTLinear wo_{nullptr};
  torch::nn::Functional act_{nullptr};
};

class T5LayerFFNImpl : public torch::nn::Module {
 public:
  T5LayerFFNImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    layer_norm_ =
        register_module("layer_norm",
                        T5LayerNorm(model_args.t5_d_model(),
                                    model_args.t5_layer_norm_epsilon()));
    if (model_args.t5_is_gated_act()) {
      dense_relu_dense_ =
          register_module("DenseReluDense",
                          std::make_shared<T5DenseGatedActDenseImpl>(context));
    } else {
      dense_relu_dense_ = register_module(
          "DenseReluDense", std::make_shared<T5DenseActDenseImpl>(context));
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor forwarded_states = layer_norm_->forward(hidden_states);
    forwarded_states = dense_relu_dense_->forward(forwarded_states);
    torch::Tensor output = hidden_states + forwarded_states;
    return output;
  }

  void load_state_dict(const StateDict& state_dict) {
    dense_relu_dense_->load_state_dict(
        state_dict.get_dict_with_prefix("DenseReluDense."));
    layer_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm."));
  }

 private:
  std::shared_ptr<T5DenseInterface> dense_relu_dense_{nullptr};
  T5LayerNorm layer_norm_{nullptr};
};
TORCH_MODULE(T5LayerFFN);

inline std::pair<std::unordered_set<int64_t>, torch::Tensor>
find_pruneable_heads_and_indices(
    const std::vector<int64_t>& heads,
    int64_t n_heads,
    int64_t head_size,
    const std::unordered_set<int64_t>& already_pruned_heads,
    torch::ScalarType dtype = torch::kBFloat16) {
  std::unordered_set<int64_t> heads_to_prune;
  for (int64_t h : heads) {
    if (already_pruned_heads.find(h) == already_pruned_heads.end()) {
      heads_to_prune.insert(h);
    }
  }
  torch::Tensor mask = torch::ones({n_heads, head_size}, dtype);

  for (int64_t head : heads_to_prune) {
    int64_t adjusted_head = head;
    for (int64_t pruned : already_pruned_heads) {
      if (pruned < head) {
        adjusted_head--;
      }
    }
    mask[adjusted_head] = 0.0f;
  }
  mask = mask.view(-1).contiguous().eq(1.0f);
  torch::Tensor index = torch::arange(mask.numel(), torch::kLong).index({mask});

  return {heads_to_prune, index};
}

class T5AttentionImpl : public torch::nn::Module {
 public:
  T5AttentionImpl(const ModelContext& context,
                  bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    has_relative_attention_bias_ = has_relative_attention_bias;
    inner_dim_ = model_args.t5_num_heads() * model_args.t5_d_kv();

    n_heads_ = model_args.t5_num_heads();
    key_value_proj_dim_ = model_args.t5_d_kv();
    d_model_ = model_args.t5_d_model();
    relative_attention_num_buckets_ =
        model_args.t5_relative_attention_num_buckets();
    relative_attention_max_distance_ =
        model_args.t5_relative_attention_max_distance();

    inner_dim_ = n_heads_ * key_value_proj_dim_;
    q_ = register_module(
        "q", xllm_ops::DiTLinear(d_model_, inner_dim_, false, options));
    k_ = register_module(
        "k", xllm_ops::DiTLinear(d_model_, inner_dim_, false, options));
    v_ = register_module(
        "v", xllm_ops::DiTLinear(d_model_, inner_dim_, false, options));
    o_ = register_module(
        "o", xllm_ops::DiTLinear(inner_dim_, d_model_, false, options));

    // q_->weight.set_data(q_->weight.to(options));
    // k_->weight.set_data(k_->weight.to(options));
    // v_->weight.set_data(v_->weight.to(options));
    // o_->weight.set_data(o_->weight.to(options));

    if (has_relative_attention_bias_) {
      relative_attention_bias_ = register_module(
          "relative_attention_bias",
          torch::nn::Embedding(relative_attention_num_buckets_, n_heads_));
    }
  }

  torch::Tensor _relative_position_bucket(torch::Tensor& relative_position,
                                          bool bidirectional = true,
                                          int64_t num_buckets = 32,
                                          int64_t max_distance = 128) const {
    torch::Tensor relative_buckets =
        torch::zeros_like(relative_position, torch::kLong);
    if (bidirectional) {
      num_buckets /= 2;
      relative_buckets +=
          (relative_position > 0).to(torch::kLong) * num_buckets;
      auto abs_relative_position = torch::abs(relative_position);
      relative_position = abs_relative_position;
    } else {
      relative_position =
          -torch::min(relative_position, torch::zeros_like(relative_position));
    }
    int64_t max_exact = num_buckets / 2;
    torch::Tensor is_small = relative_position < max_exact;
    auto relative_position_float = relative_position.to(torch::kFloat);
    auto max_exact_float = static_cast<float>(max_exact);
    auto max_distance_float = static_cast<float>(max_distance);
    torch::Tensor relative_position_if_large =
        max_exact + (torch::log(relative_position_float / max_exact_float) /
                     std::log(max_distance_float / max_exact_float) *
                     (num_buckets - max_exact))
                        .to(torch::kLong);
    relative_position_if_large = torch::min(
        relative_position_if_large,
        torch::full_like(
            relative_position_if_large, num_buckets - 1, torch::kLong));
    relative_buckets +=
        torch::where(is_small, relative_position, relative_position_if_large);
    return relative_buckets;
  }

  torch::Tensor compute_bias(
      int64_t query_length,
      int64_t key_length,
      std::optional<torch::Device> device = std::nullopt,
      const std::optional<torch::Tensor>& cache_position = std::nullopt) const {
    if (!has_relative_attention_bias_) {
      return torch::zeros(
          {1, n_heads_, query_length, key_length},
          torch::dtype(torch::kFloat).device(device.value_or(torch::kCPU)));
    }

    torch::Device dev =
        device.value_or(relative_attention_bias_->weight.device());

    torch::Tensor context_position;
    if (cache_position.has_value()) {
      context_position = cache_position.value().unsqueeze(1).to(dev);
    } else {
      context_position =
          torch::arange(query_length, torch::dtype(torch::kLong).device(dev))
              .unsqueeze(1);
    }

    torch::Tensor memory_position =
        torch::arange(key_length, torch::dtype(torch::kLong).device(dev))
            .unsqueeze(0);
    torch::Tensor relative_position = memory_position - context_position;
    torch::Tensor relative_position_bucket =
        _relative_position_bucket(relative_position,
                                  true,
                                  relative_attention_num_buckets_,
                                  relative_attention_max_distance_);
    torch::Tensor values =
        const_cast<torch::nn::EmbeddingImpl*>(relative_attention_bias_.get())
            ->forward(relative_position_bucket);
    values = values.permute({2, 0, 1}).unsqueeze(0);

    return values;
  }

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& mask = std::nullopt,
      const std::optional<torch::Tensor>& key_value_states = std::nullopt,
      const std::optional<torch::Tensor>& position_bias = std::nullopt,
      const std::optional<torch::Tensor>& layer_head_mask = std::nullopt,
      bool output_attentions = false) {
    int64_t batch_size = hidden_states.size(0);
    int64_t seq_length = hidden_states.size(1);
    bool is_cross_attention = key_value_states.has_value();
    torch::Tensor query_states = q_->forward(hidden_states);
    query_states =
        query_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, seq_len, head_dim)

    torch::Tensor current_states =
        is_cross_attention ? key_value_states.value() : hidden_states;
    torch::Tensor key_states = k_->forward(current_states);
    torch::Tensor value_states = v_->forward(current_states);
    key_states =
        key_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, key_len, head_dim)
    value_states =
        value_states.view({batch_size, -1, n_heads_, key_value_proj_dim_})
            .transpose(1, 2);  // (batch_size, n_heads, key_len, head_dim)
    torch::Tensor scores = torch::matmul(
        query_states,
        key_states.transpose(3, 2));  // (batch, n_heads, seq_len, key_len)
    torch::Tensor curr_position_bias;
    if (position_bias.has_value() && position_bias.value().numel() > 0) {
      curr_position_bias = position_bias.value();
    } else {
      int64_t key_length = key_states.size(-2);
      if (!has_relative_attention_bias_) {
        curr_position_bias =
            torch::zeros({1, n_heads_, seq_length, key_length},
                         torch::dtype(scores.dtype()).device(scores.device()));
      } else {
        torch::Tensor bias =
            compute_bias(seq_length, key_length, scores.device());
        curr_position_bias = bias.index(
            {torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(-seq_length, torch::indexing::None),
             torch::indexing::Slice()});
      }
      if (mask.has_value() && mask.value().numel() > 0) {
        torch::Tensor causal_mask = mask.value().index(
            {torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(0, key_states.size(-2))});
        curr_position_bias = curr_position_bias + causal_mask;
      }
    }
    if (!pruned_heads_.empty()) {
      torch::Tensor head_mask =
          torch::ones(n_heads_ + pruned_heads_.size(), torch::kBool)
              .to(scores.device());
      for (int64_t pruned : pruned_heads_) {
        head_mask[pruned] = false;
      }
      curr_position_bias = curr_position_bias.index({torch::indexing::Slice(),
                                                     head_mask,
                                                     torch::indexing::Slice(),
                                                     torch::indexing::Slice()});
    }
    scores += curr_position_bias;
    torch::Tensor attn_weights =
        torch::softmax(scores.to(torch::kFloat), -1).to(scores.dtype());
    if (layer_head_mask.has_value() && layer_head_mask.value().numel() > 0) {
      attn_weights = attn_weights * layer_head_mask.value();
    }
    torch::Tensor attn_output = torch::matmul(
        attn_weights, value_states);  // (batch, n_heads, seq_len, head_dim)
    attn_output = attn_output.transpose(1, 2)
                      .contiguous();  // (batch, seq_len, n_heads, head_dim)
    attn_output = attn_output.view({batch_size, -1, inner_dim_});
    attn_output = o_->forward(attn_output);
    std::vector<torch::Tensor> outputs = {attn_output, curr_position_bias};
    if (output_attentions) {
      outputs.push_back(attn_weights);
    }
    return outputs;
  }

  void load_state_dict(const StateDict& state_dict) {
    q_->load_state_dict(state_dict.get_dict_with_prefix("q."));
    k_->load_state_dict(state_dict.get_dict_with_prefix("k."));
    v_->load_state_dict(state_dict.get_dict_with_prefix("v."));
    o_->load_state_dict(state_dict.get_dict_with_prefix("o."));
    auto relative_attention_bias_weight_ =
        state_dict.get_tensor("relative_attention_bias.weight");
    if (relative_attention_bias_weight_.defined()) {
      DCHECK_EQ(relative_attention_bias_->weight.sizes(),
                relative_attention_bias_weight_.sizes())
          << "relative_attention_bias weight size mismatch: expected "
          << relative_attention_bias_->weight.sizes() << " but got "
          << relative_attention_bias_weight_.sizes();
      relative_attention_bias_->weight.data().copy_(
          relative_attention_bias_weight_);
    }
  }

 private:
  bool has_relative_attention_bias_;
  int64_t relative_attention_num_buckets_;
  int64_t relative_attention_max_distance_;
  int64_t d_model_;
  int64_t key_value_proj_dim_;
  int64_t n_heads_;
  int64_t inner_dim_;
  std::optional<int64_t> layer_idx_;
  xllm_ops::DiTLinear q_{nullptr};
  xllm_ops::DiTLinear k_{nullptr};
  xllm_ops::DiTLinear v_{nullptr};
  xllm_ops::DiTLinear o_{nullptr};
  torch::nn::Embedding relative_attention_bias_{nullptr};
  std::unordered_set<int64_t> pruned_heads_;
};
TORCH_MODULE(T5Attention);

class T5LayerSelfAttentionImpl : public torch::nn::Module {
 public:
  T5LayerSelfAttentionImpl(const ModelContext& context,
                           bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attention_ = register_module(
        "SelfAttention", T5Attention(context, has_relative_attention_bias));
    layer_norm_ =
        register_module("layer_norm",
                        T5LayerNorm(model_args.t5_d_model(),
                                    model_args.t5_layer_norm_epsilon()));
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("SelfAttention."));
    layer_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("layer_norm."));
  }

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt,
      const std::optional<torch::Tensor>& position_bias = std::nullopt,
      const std::optional<torch::Tensor>& layer_head_mask = std::nullopt,
      bool output_attentions = false) {
    torch::Tensor normed_hidden_states = layer_norm_->forward(hidden_states);
    auto attention_output = self_attention_->forward(normed_hidden_states,
                                                     attention_mask,
                                                     std::nullopt,
                                                     position_bias,
                                                     layer_head_mask,
                                                     output_attentions);
    torch::Tensor updated_hidden_states = hidden_states + attention_output[0];
    // hidden_states, position_bias, [attn_weights])
    std::vector<torch::Tensor> outputs = {updated_hidden_states};
    outputs.push_back(attention_output[1]);
    if (output_attentions && attention_output.size() > 2) {
      outputs.push_back(attention_output[2]);
    }
    return outputs;
  }

 private:
  T5Attention self_attention_{nullptr};
  T5LayerNorm layer_norm_{nullptr};
};
TORCH_MODULE(T5LayerSelfAttention);

class T5BlockImpl : public torch::nn::Module {
 public:
  T5BlockImpl(const ModelContext& context, bool has_relative_attention_bias) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    self_attention_ = register_module(
        "SelfAttention",
        T5LayerSelfAttention(context, has_relative_attention_bias));
    ff_layer_ = register_module("LayerFFN", T5LayerFFN(context));
  }

  std::vector<torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& attention_mask = std::nullopt,
      const std::optional<torch::Tensor>& position_bias = std::nullopt,
      const std::optional<torch::Tensor>& layer_head_mask = std::nullopt,
      bool output_attentions = false) {
    auto self_attention_outputs = self_attention_->forward(hidden_states,
                                                           attention_mask,
                                                           position_bias,
                                                           layer_head_mask,
                                                           output_attentions);
    torch::Tensor curr_hidden_states = self_attention_outputs[0];
    std::vector<torch::Tensor> attention_outputs;
    for (size_t i = 1; i < self_attention_outputs.size(); ++i) {
      attention_outputs.push_back(self_attention_outputs[i]);
    }
    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }
    curr_hidden_states = ff_layer_->forward(curr_hidden_states);
    if (curr_hidden_states.dtype() == torch::kFloat16) {
      curr_hidden_states = clamp_inf_values(curr_hidden_states);
    }
    std::vector<torch::Tensor> outputs = {curr_hidden_states};
    outputs.insert(
        outputs.end(), attention_outputs.begin(), attention_outputs.end());
    return outputs;
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("layer.0."));
    ff_layer_->load_state_dict(state_dict.get_dict_with_prefix("layer.1."));
  }

 private:
  torch::Tensor clamp_inf_values(const torch::Tensor& x) const {
    float max_val;
    if (x.scalar_type() == torch::kFloat16) {
      max_val = 65504.0f;
    } else if (x.scalar_type() == torch::kFloat32) {
      max_val = std::numeric_limits<float>::max();
    } else if (x.scalar_type() == torch::kBFloat16) {
      max_val = 3.3895313892515355e+38f;
    } else {
      max_val = std::numeric_limits<float>::max();
    }
    torch::Tensor clamp_value =
        torch::where(torch::isinf(x).any(),
                     torch::tensor(max_val - 1000.0f, x.options()),
                     torch::tensor(max_val, x.options()));

    return torch::clamp(x, -clamp_value, clamp_value);
  }
  T5LayerSelfAttention self_attention_{nullptr};
  T5LayerFFN ff_layer_{nullptr};
};
TORCH_MODULE(T5Block);

class T5EncoderModelImpl : public torch::nn::Module {
 public:
  T5EncoderModelImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    embed_tokens_ =
        register_module("embed_tokens",
                        torch::nn::Embedding(model_args.t5_vocab_size(),
                                             model_args.t5_d_model()));
    embed_tokens_->weight.set_data(embed_tokens_->weight.to(options));
    for (int64_t i = 0; i < model_args.t5_num_layers(); ++i) {
      bool has_relative_bias = (i == 0);
      blocks_.push_back(register_module("block_" + std::to_string(i),
                                        T5Block(context, has_relative_bias)));
    }
    final_layer_norm_ =
        register_module("final_layer_norm",
                        T5LayerNorm(model_args.t5_d_model(),
                                    model_args.t5_layer_norm_epsilon()));
  }

  torch::nn::Embedding& get_input_embeddings() { return embed_tokens_; }
  void set_input_embeddings(const torch::nn::Embedding& new_embeddings) {
    embed_tokens_ = new_embeddings;
  }

  torch::Tensor forward(const torch::Tensor& input_ids) {
    // prepare input parameters
    // input parameters
    // input_ids
    auto options = torch::TensorOptions()
                       .dtype(torch::typeMetaToScalarType(input_ids.dtype()))
                       .device(input_ids.device());
    bool output_hidden_states = false;
    bool output_attentions = false;
    std::optional<torch::Tensor> attention_mask = std::nullopt;
    std::optional<torch::Tensor> head_mask = std::nullopt;
    torch::Tensor hidden_states = embed_tokens_->forward(input_ids);
    auto input_shape =
        hidden_states.sizes();  // (batch_size, seq_length, d_model)
    int64_t batch_size = input_shape[0];
    int64_t seq_length = input_shape[1];
    torch::Tensor causal_mask;
    if (attention_mask.has_value()) {
      causal_mask =
          attention_mask.value().unsqueeze(1).unsqueeze(1).to(options);
      causal_mask = (1.0 - causal_mask) * std::numeric_limits<float>::lowest();
    } else {
      causal_mask = torch::Tensor();
    }
    std::vector<torch::Tensor> all_hidden_states;
    std::vector<torch::Tensor> all_attentions;
    if (output_hidden_states) {
      all_hidden_states.push_back(hidden_states);
    }
    torch::Tensor position_bias = torch::Tensor();
    for (size_t i = 0; i < blocks_.size(); ++i) {
      torch::Tensor layer_head_mask;
      if (head_mask.has_value()) {
        layer_head_mask =
            head_mask.value().index({torch::tensor((int64_t)i)}).to(options);
      } else {
        layer_head_mask = torch::Tensor();
      }
      auto layer_outputs = blocks_[i]->forward(hidden_states,
                                               causal_mask,
                                               position_bias,
                                               layer_head_mask,
                                               output_attentions);
      hidden_states = layer_outputs[0];
      position_bias = layer_outputs[1];
      if (output_hidden_states) {
        all_hidden_states.push_back(hidden_states);
      }
      if (output_attentions && layer_outputs.size() > 2) {
        all_attentions.push_back(layer_outputs[2]);
      }

      layer_outputs.clear();
    }
    hidden_states = final_layer_norm_->forward(hidden_states);
    if (output_hidden_states) {
      all_hidden_states.push_back(hidden_states);
    }
    std::vector<torch::Tensor> outputs = {hidden_states};
    if (output_hidden_states) {
      outputs.push_back(
          torch::stack(all_hidden_states,
                       1));  // (batch_size, num_layers, seq_length, d_model)
    }
    if (output_attentions) {
      outputs.push_back(torch::stack(
          all_attentions,
          1));  // (batch_size, num_layers, n_heads, seq_length, seq_length)
    }
    return outputs[0];
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      const auto embedding_weight = state_dict->get_tensor("shared.weight");
      if (embedding_weight.defined()) {
        DCHECK_EQ(embedding_weight.sizes(), embed_tokens_->weight.sizes())
            << "Embedding weight size mismatch: expected "
            << embed_tokens_->weight.sizes() << ", got "
            << embedding_weight.sizes();
        embed_tokens_->weight.data().copy_(embedding_weight);
      }
      const auto final_layer_norm_weight =
          state_dict->get_tensor("encoder.final_layer_norm.weight");
      if (final_layer_norm_weight.defined()) {
        DCHECK_EQ(final_layer_norm_weight.sizes(),
                  final_layer_norm_->weight.sizes())
            << "Final layer norm weight size mismatch: expected "
            << final_layer_norm_->weight.sizes() << ", got "
            << final_layer_norm_weight.sizes();
        final_layer_norm_->weight.data().copy_(final_layer_norm_weight);
      }
      for (int64_t i = 0; i < blocks_.size(); ++i) {
        const auto block_prefix = "encoder.block." + std::to_string(i) + ".";
        blocks_[i]->load_state_dict(
            state_dict->get_dict_with_prefix(block_prefix));
      }
    }
    LOG(INFO) << "T5EncoderModel loaded successfully.";
  }

 private:
  T5LayerNorm final_layer_norm_{nullptr};
  torch::nn::Embedding embed_tokens_{nullptr};
  std::vector<T5Block> blocks_;
};
TORCH_MODULE(T5EncoderModel);

REGISTER_MODEL_ARGS(T5EncoderModel, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(model_type, "model_type", "t5encoder");
  LOAD_ARG_OR(t5_vocab_size, "vocab_size", 32128);
  LOAD_ARG_OR(t5_d_model, "d_model", 4096);
  LOAD_ARG_OR(t5_num_layers, "num_layers", 24);
  LOAD_ARG_OR(t5_d_kv, "d_kv", 64);
  LOAD_ARG_OR(t5_num_heads, "num_heads", 64);
  LOAD_ARG_OR(t5_d_ff, "d_ff", 10240);
  LOAD_ARG_OR(t5_dense_act_fn, "dense_act_fn", "gelu_new");
  LOAD_ARG_OR(t5_is_gated_act, "is_gated_act", true);
  LOAD_ARG_OR(
      t5_relative_attention_num_buckets, "relative_attention_num_buckets", 32);
  LOAD_ARG_OR(t5_relative_attention_max_distance,
              "relative_attention_max_distance",
              128);
  LOAD_ARG_OR(t5_layer_norm_epsilon, "layer_norm_epsilon", 1e-6f);
});
}  // namespace xllm
