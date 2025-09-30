#pragma once
#include <glog/logging.h>
#include <torch/nn/functional/linear.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "framework/model_context.h"
#include "kernels/npu/xllm_ops/add_batch_matmul.h"
#include "kernels/npu/xllm_ops/add_matmul.h"
#include "kernels/npu/xllm_ops/rms_norm.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "xllm/core/kernels/npu/xllm_ops/add_matmul.h"
// DiT model compatible with huggingface weights
//   ref to:
//   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
namespace xllm {

struct AllToAll4DHandle {
  torch::Tensor mid;  // branch A: (P, shard_seqlen, bs, shard_hc, hs)
                      // branch B: (P, shard_hc,     shard_seqlen, bs, hs)

  int64_t bs = 0;
  int64_t seqlen = 0;    // branch A
  int64_t shard_hc = 0;  // branch A
  int64_t hs = 0;

  int64_t shard_seqlen = 0;  // branch B
  int64_t hc = 0;            // branch B

  int64_t gather_idx = 0;
  int64_t gather_pad = 0;
  bool use_post2 = false;

  std::shared_ptr<c10_npu::NPUEvent> done_event;
  bool is_async = false;
};

inline AllToAll4DHandle all_to_all_4D(const torch::Tensor& input_,
                                      int rank,
                                      int world_size,
                                      int scatter_idx,
                                      int gather_idx,
                                      bool is_sync,
                                      ParallelArgs pg) {
  TORCH_CHECK(input_.dim() == 4,
              "all_to_all_4D: input must be 4D, got dim=",
              input_.dim());
  auto input = input_.contiguous();
  const int P = world_size;
  const int r = rank;
  AllToAll4DHandle h;
  h.gather_idx = gather_idx;
  h.gather_pad = 0;

  if (scatter_idx == 2 && gather_idx == 1) {
    // branch A : from "sequence shard" -> "head shard"
    // input: (bs, shard_seqlen, hc, hs)  output (bs, seqlen, hc/P, hs)
    auto sizes = input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t shard_seqlen = sizes[1];
    const int64_t hc = sizes[2];
    const int64_t hs = sizes[3];
    TORCH_CHECK(hc % P == 0,
                "all_to_all_4D(A): hc must be divisible by world_size");
    const int64_t shard_hc = hc / P;

    // prepare expected shape for All2All (P, shard_seqlen, bs, shard_hc, hs)
    auto input_t = input.reshape({bs, shard_seqlen, P, shard_hc, hs})
                       .transpose(0, 2)  // (P, shard_seqlen, bs, shard_hc, hs)
                       .contiguous();

    auto send_flat = input_t.reshape({P, shard_seqlen * bs * shard_hc * hs})
                         .contiguous()
                         .reshape({P * shard_seqlen * bs * shard_hc * hs});

#if defined(USE_NPU)
    auto recv_flat = parallel_state::all_to_all_equal(send_flat, is_sync, pg);
    auto stream = c10_npu::getCurrentNPUStream();
    std::shared_ptr<c10_npu::NPUEvent> ev;
    if (!is_sync) {
      ev = std::make_shared<c10_npu::NPUEvent>();
      ev->record(stream);
    }
#endif
    auto mid =
        recv_flat.reshape({P, shard_seqlen, bs, shard_hc, hs}).contiguous();
    h.mid = mid;
    h.bs = bs;
    h.seqlen = shard_seqlen * P;
    h.shard_hc = shard_hc;
    h.hs = hs;
    h.use_post2 = true;
    h.done_event = ev;
    h.is_async = !is_sync;
    return h;

  } else if (scatter_idx == 1 && gather_idx == 2) {
    // branch B : from "head shard" -> "sequence shard"
    // input: (bs, seqlen, hc/P, hs)  output (bs, seqlen/P, hc, hs)
    auto sizes = input.sizes().vec();
    const int64_t bs = sizes[0];
    const int64_t seqlen = sizes[1];
    const int64_t shard_hc = sizes[2];
    const int64_t hs = sizes[3];
    TORCH_CHECK(seqlen % P == 0,
                "all_to_all_4D(B): seqlen must be divisible by world_size");
    const int64_t shard_seqlen = seqlen / P;
    const int64_t hc = shard_hc * P;

    // prepare expected shape for All2All (P, shard_hc, shard_seqlen, bs, hs)
    auto input_t = input.reshape({bs, P, shard_seqlen, shard_hc, hs})
                       .transpose(0, 3)  // (shard_hc, P, shard_seqlen, bs, hs)
                       .transpose(0, 1)  // (P, shard_hc, shard_seqlen, bs, hs)
                       .contiguous();

    auto send_flat =
        input_t.reshape({P * shard_hc * shard_seqlen * bs * hs}).contiguous();
#if defined(USE_NPU)
    auto recv_flat = parallel_state::all_to_all_equal(send_flat, is_sync, pg);
    auto stream = c10_npu::getCurrentNPUStream();
    std::shared_ptr<c10_npu::NPUEvent> ev;
    if (!is_sync) {
      ev = std::make_shared<c10_npu::NPUEvent>();
      ev->record(stream);
    }
#endif
    auto mid = recv_flat.reshape({P, shard_hc, shard_seqlen, bs, hs})
                   .contiguous();  // (P, shard_hc, shard_seqlen, bs, hs)
    h.mid = mid;
    h.bs = bs;
    h.hc = hc;
    h.shard_seqlen = shard_seqlen;
    h.hs = hs;
    h.use_post2 = false;
    h.done_event = ev;
    h.is_async = !is_sync;
    return h;

  } else {
    TORCH_CHECK(false,
                "all_to_all_4D: only (scatter_idx,gather_idx)=(2,1) or (1,2) "
                "are supported");
  }
}

// branch A post processing ： (P, shard_seqlen, bs, shard_hc, hs)
// → (seqlen, bs, shard_hc, hs) → (bs, seqlen, shard_hc, hs)
inline torch::Tensor all_to_all_4D_post2(const AllToAll4DHandle& h) {
  TORCH_CHECK(h.use_post2, "all_to_all_4D_post2: handle not from (2->1) path");
  if (h.is_async && h.done_event) {
    h.done_event->synchronize();
  }
  auto out = h.mid.reshape({h.seqlen, h.bs, h.shard_hc, h.hs})
                 .transpose(0, 1)  // (bs, seqlen, shard_hc, hs)
                 .contiguous();
  if (h.gather_pad > 0) {
    out = out.narrow(h.gather_idx, 0, out.size(h.gather_idx) - h.gather_pad);
  }
  return out;
}

// branch B post processing： (P, shard_hc, shard_seqlen, bs, hs)
// → (hc, shard_seqlen, bs, hs) → (bs, shard_seqlen, hc, hs)
inline torch::Tensor all_to_all_4D_post(const AllToAll4DHandle& h) {
  TORCH_CHECK(!h.use_post2, "all_to_all_4D_post: handle not from (1->2) path");
  if (h.is_async && h.done_event) {
    h.done_event->synchronize();
  }
  auto out = h.mid.reshape({h.hc, h.shard_seqlen, h.bs, h.hs})
                 .transpose(0, 2)  // (bs, shard_seqlen, hc, hs)
                 .contiguous();
  if (h.gather_pad > 0) {
    out = out.narrow(h.gather_idx, 0, out.size(h.gather_idx) - h.gather_pad);
  }
  return out;
}

inline torch::Tensor gather_sequence(const torch::Tensor& input_,
                                     int world_size,
                                     int64_t dim,
                                     int64_t pad,
                                     ParallelArgs pg) {
  auto input = input_.contiguous();
  if (world_size == 1) {
    return input;
  }

  // all gather
  auto tensor_list = parallel_state::all_gather(input, pg);

  // concat
  auto output = torch::cat(tensor_list, dim);

  if (pad > 0) {
    output = output.narrow(dim, 0, output.size(dim) - pad);
  }

  return output;
}

inline torch::Tensor split_sequence(const torch::Tensor& input_,
                                    int64_t world_size,
                                    int64_t rank,
                                    int64_t dim,
                                    int64_t pad) {
  if (world_size == 1) {
    return input_;
  }

  torch::Tensor input = input_;
  if (pad > 0) {
    std::vector<int64_t> pad_size(input.sizes().begin(), input.sizes().end());
    pad_size[dim] = pad;
    input = torch::cat(
        {input,
         torch::zeros(pad_size,
                      torch::dtype(input.dtype()).device(input.device()))},
        dim);
  }

  int64_t dim_size = input.size(dim);

  auto tensor_list = torch::split(input, dim_size / world_size, dim);
  auto output = tensor_list[rank].contiguous();
  return output;
}

inline torch::Tensor apply_rotary_emb(const torch::Tensor& x,
                                      const torch::Tensor& freqs_cis,
                                      bool head_first = true) {
  // assume freqs_cis is [2, S, D]，[0] is cos，[1] is sin
  torch::Tensor cos;
  torch::Tensor sin;
  if (head_first) {
    cos = freqs_cis[0].unsqueeze(0).unsqueeze(1).to(
        torch::kBFloat16);  // [1, 1, 6542, 128]
    sin = freqs_cis[1].unsqueeze(0).unsqueeze(1).to(
        torch::kBFloat16);  // [1, 1, 6542, 128]
  } else {
    cos = freqs_cis[0].unsqueeze(0).unsqueeze(2).to(
        torch::kBFloat16);  // [1, 6542, 1, 128]
    sin = freqs_cis[1].unsqueeze(0).unsqueeze(2).to(
        torch::kBFloat16);  // [1, 6542, 1, 128]
  }

  // std::vector<int64_t> reshape_shape;
  // for (int64_t i = 0; i < x.dim() - 1; ++i) {
  //   reshape_shape.push_back(x.size(i));
  // }
  // reshape_shape.push_back(-1);
  // reshape_shape.push_back(2);
  // torch::Tensor reshaped = x.reshape(reshape_shape);

  // torch::Tensor x_real = reshaped.select(-1, 0);
  // torch::Tensor x_imag = reshaped.select(-1, 1);
  //  x_rotated = [-x_imag, x_real]
  // torch::Tensor neg_x_imag = -x_imag;
  // auto x_rotated = torch::stack({neg_x_imag, x_real}, -1).flatten(3);
  // return (x.to(torch::kFloat32) * cos.to(torch::kFloat32) +
  //         x_rotated.to(torch::kFloat32) * sin.to(torch::kFloat32))
  //     .to(x.dtype());
  return at_npu::native::custom_ops::npu_rope(x, cos, sin, 1);
}

class FluxSingleAttentionImpl : public torch::nn::Module {
 private:
  xllm_ops::DiTLinear to_q_{nullptr};
  xllm_ops::DiTLinear to_k_{nullptr};
  xllm_ops::DiTLinear to_v_{nullptr};
  int64_t heads_;
  xllm_ops::RMSNorm norm_q_{nullptr};
  xllm_ops::RMSNorm norm_k_{nullptr};
  torch::TensorOptions options_;
  // For sequence parallel
  int world_size_{1};
  int rank_{0};
  bool use_sp_{false};
  ParallelArgs pg_;

 public:
  void load_state_dict(const StateDict& state_dict) {
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    // to_q
    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));
  }

  FluxSingleAttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()),
        pg_(context.get_parallel_args()) {
    auto model_args = context.get_model_args();
    world_size_ = pg_.world_size();
    rank_ = pg_.rank();
    use_sp_ = world_size_ > 1;
    heads_ = model_args.dit_num_attention_heads();
    auto head_dim = model_args.dit_attention_head_dim();
    auto query_dim = heads_ * head_dim;
    auto out_dim = query_dim;
    to_q_ = register_module(
        "to_q",
        xllm_ops::DiTLinear(query_dim, out_dim, true /*has_bias*/, options_));
    to_k_ = register_module(
        "to_k",
        xllm_ops::DiTLinear(query_dim, out_dim, true /*has_bias*/, options_));
    to_v_ = register_module(
        "to_v",
        xllm_ops::DiTLinear(query_dim, out_dim, true /*has_bias*/, options_));

    norm_q_ = register_module(
        "norm_q", xllm_ops::RMSNorm(head_dim, 1e-6f, true, false, options_));
    norm_k_ = register_module(
        "norm_k", xllm_ops::RMSNorm(head_dim, 1e-6f, true, false, options_));
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cmo_matmul_all2all(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      int64_t attn_heads,
      int64_t head_dim) {
    int64_t batch_size = encoder_hidden_states.size(0);
    torch::Tensor query = to_q_->forward(hidden_states);
    auto handle_q =
        all_to_all_4D(query.view({batch_size, -1, attn_heads, head_dim}),
                      rank_,
                      world_size_,
                      2,
                      1,
                      false,
                      pg_);
    torch::Tensor key = to_k_->forward(encoder_hidden_states);
    auto handle_k =
        all_to_all_4D(key.view({batch_size, -1, attn_heads, head_dim}),
                      rank_,
                      world_size_,
                      2,
                      1,
                      false,
                      pg_);
    torch::Tensor value = to_v_->forward(encoder_hidden_states);
    auto handle_v =
        all_to_all_4D(value.view({batch_size, -1, attn_heads, head_dim}),
                      rank_,
                      world_size_,
                      2,
                      1,
                      false,
                      pg_);
    query = all_to_all_4D_post2(handle_q);
    key = all_to_all_4D_post2(handle_k);
    value = all_to_all_4D_post2(handle_v);
    return std::make_tuple(query, key, value);
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& image_rotary_emb) {
    int64_t batch_size, channel, height, width;

    // Reshape 4D input to [B, seq_len, C]
    torch::Tensor hidden_states_ =
        hidden_states;  // Use copy to avoid modifying input
    batch_size = hidden_states_.size(0);
    int64_t inner_dim = to_k_->weight_.size(0);
    int64_t attn_heads = heads_;
    int64_t head_dim = inner_dim / attn_heads;
    // Self-attention: use hidden_states as context
    torch::Tensor context = hidden_states_;
    torch::Tensor query, key, value;
    if (use_sp_) {
      std::tie(query, key, value) =
          cmo_matmul_all2all(hidden_states_, context, attn_heads, head_dim);
    } else {
      // Compute QKV projections
      query = to_q_->forward(hidden_states_);
      key = to_k_->forward(context);
      value = to_v_->forward(context);

      // Reshape for multi-head attention
      query = query.view({batch_size, -1, attn_heads, head_dim}).contiguous();
      key = key.view({batch_size, -1, attn_heads, head_dim}).contiguous();
      value = value.view({batch_size, -1, attn_heads, head_dim}).contiguous();
    }

    // Apply Q/K normalization if enabled
    if (norm_q_) query = norm_q_->forward(query);
    if (norm_k_) key = norm_k_->forward(key);
    // Apply rotary positional embedding
    query = apply_rotary_emb(query, image_rotary_emb, false);
    key = apply_rotary_emb(key, image_rotary_emb, false);
    // Compute scaled dot-product attention (no mask, no dropout)
    // torch::Tensor attn_output = torch::scaled_dot_product_attention(
    //    query, key, value, torch::nullopt, 0.0, false);
    int64_t head_num_ = query.size(2);
    int64_t head_dim_ = query.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(query,
                                                         key,
                                                         value,
                                                         head_num_,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);
    attn_output = attn_output.to(query.dtype());
    if (use_sp_) {
      attn_heads = heads_ / world_size_;
      attn_output =
          attn_output.view({batch_size, -1, attn_heads, head_dim}).contiguous();
      auto handle =
          all_to_all_4D(attn_output, rank_, world_size_, 1, 2, true, pg_);
      attn_output = all_to_all_4D_post(handle);
      attn_output = attn_output.view({batch_size, -1, inner_dim}).contiguous();
      return attn_output;
    }
    return attn_output.flatten(2);
  }
};
TORCH_MODULE(FluxSingleAttention);

class FluxAttentionImpl : public torch::nn::Module {
 private:
  xllm_ops::DiTLinear to_q_{nullptr};
  xllm_ops::DiTLinear to_k_{nullptr};
  xllm_ops::DiTLinear to_v_{nullptr};
  xllm_ops::DiTLinear add_q_proj_{nullptr};
  xllm_ops::DiTLinear add_k_proj_{nullptr};
  xllm_ops::DiTLinear add_v_proj_{nullptr};
  xllm_ops::DiTLinear to_out_{nullptr};
  xllm_ops::DiTLinear to_add_out_{nullptr};

  xllm_ops::RMSNorm norm_q_{nullptr};
  xllm_ops::RMSNorm norm_k_{nullptr};
  xllm_ops::RMSNorm norm_added_q_{nullptr};
  xllm_ops::RMSNorm norm_added_k_{nullptr};
  int64_t heads_;
  torch::TensorOptions options_;
  // For sequence parallel
  int world_size_{1};
  int rank_{0};
  bool use_sp_{false};
  ParallelArgs pg_;

 public:
  void load_state_dict(const StateDict& state_dict) {
    to_q_->load_state_dict(state_dict.get_dict_with_prefix("to_q."));
    //  to_k
    to_k_->load_state_dict(state_dict.get_dict_with_prefix("to_k."));
    //  to_v
    to_v_->load_state_dict(state_dict.get_dict_with_prefix("to_v."));
    //  to_out
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));
    //  to_add_out
    to_add_out_->load_state_dict(
        state_dict.get_dict_with_prefix("to_add_out."));
    //  add_q_proj
    add_q_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_q_proj."));
    //  add_k_proj
    add_k_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_k_proj."));
    //  add_v_proj
    add_v_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("add_v_proj."));
    // norm_q
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    // norm_k
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
    // norm_added_q
    norm_added_q_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_q."));
    // norm_added_k
    norm_added_k_->load_state_dict(
        state_dict.get_dict_with_prefix("norm_added_k."));
  }

  FluxAttentionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()),
        pg_(context.get_parallel_args()) {
    auto model_args = context.get_model_args();
    world_size_ = pg_.world_size();
    rank_ = pg_.rank();
    use_sp_ = world_size_ > 1;
    heads_ = model_args.dit_num_attention_heads();
    auto head_dim = model_args.dit_attention_head_dim();
    auto query_dim = heads_ * head_dim;
    auto out_dim = query_dim;
    auto added_kv_proj_dim = query_dim;

    to_q_ = register_module(
        "to_q", xllm_ops::DiTLinear(query_dim, out_dim, true, options_));
    to_k_ = register_module(
        "to_k", xllm_ops::DiTLinear(query_dim, out_dim, true, options_));
    to_v_ = register_module(
        "to_v", xllm_ops::DiTLinear(query_dim, out_dim, true, options_));
    add_q_proj_ = register_module(
        "add_q_proj",
        xllm_ops::DiTLinear(added_kv_proj_dim, out_dim, true, options_));

    add_k_proj_ = register_module(
        "add_k_proj",
        xllm_ops::DiTLinear(added_kv_proj_dim, out_dim, true, options_));
    add_v_proj_ = register_module(
        "add_v_proj",
        xllm_ops::DiTLinear(added_kv_proj_dim, out_dim, true, options_));

    to_out_ = register_module(
        "to_out",
        xllm_ops::DiTLinear(added_kv_proj_dim, out_dim, true, options_));

    to_add_out_ = register_module(
        "to_add_out",
        xllm_ops::DiTLinear(out_dim, added_kv_proj_dim, true, options_));

    norm_q_ = register_module(
        "norm_q", xllm_ops::RMSNorm(head_dim, 1e-6f, true, false, options_));
    norm_k_ = register_module(
        "norm_k", xllm_ops::RMSNorm(head_dim, 1e-6f, true, false, options_));
    norm_added_q_ = register_module(
        "norm_added_q",
        xllm_ops::RMSNorm(head_dim, 1e-6f, true, false, options_));
    norm_added_k_ = register_module(
        "norm_added_k",
        xllm_ops::RMSNorm(head_dim, 1e-6f, true, false, options_));
  }

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  cmo_matmul_all2all(const torch::Tensor& hidden_states_reshaped,
                     const torch::Tensor& encoder_hidden_states_reshaped,
                     int64_t attn_heads,
                     int64_t head_dim) {
    int64_t batch_size = encoder_hidden_states_reshaped.size(0);
    torch::Tensor query = to_q_->forward(hidden_states_reshaped);
    auto handle_q =
        all_to_all_4D(query.view({batch_size, -1, attn_heads, head_dim}),
                      rank_,
                      world_size_,
                      2,
                      1,
                      false,
                      pg_);
    torch::Tensor key = to_k_->forward(hidden_states_reshaped);
    auto handle_k =
        all_to_all_4D(key.view({batch_size, -1, attn_heads, head_dim}),
                      rank_,
                      world_size_,
                      2,
                      1,
                      false,
                      pg_);
    torch::Tensor value = to_v_->forward(hidden_states_reshaped);
    auto handle_v =
        all_to_all_4D(value.view({batch_size, -1, attn_heads, head_dim}),
                      rank_,
                      world_size_,
                      2,
                      1,
                      false,
                      pg_);
    torch::Tensor encoder_hidden_states_query_proj =
        add_q_proj_->forward(encoder_hidden_states_reshaped);
    torch::Tensor encoder_hidden_states_key_proj =
        add_k_proj_->forward(encoder_hidden_states_reshaped);
    torch::Tensor encoder_hidden_states_value_proj =
        add_v_proj_->forward(encoder_hidden_states_reshaped);
    auto handle_eq = all_to_all_4D(encoder_hidden_states_query_proj.view(
                                       {batch_size, -1, attn_heads, head_dim}),
                                   rank_,
                                   world_size_,
                                   2,
                                   1,
                                   false,
                                   pg_);
    auto handle_ek = all_to_all_4D(encoder_hidden_states_key_proj.view(
                                       {batch_size, -1, attn_heads, head_dim}),
                                   rank_,
                                   world_size_,
                                   2,
                                   1,
                                   false,
                                   pg_);
    auto handle_ev = all_to_all_4D(encoder_hidden_states_value_proj.view(
                                       {batch_size, -1, attn_heads, head_dim}),
                                   rank_,
                                   world_size_,
                                   2,
                                   1,
                                   false,
                                   pg_);
    query = all_to_all_4D_post2(handle_q);
    key = all_to_all_4D_post2(handle_k);
    if (norm_q_) query = norm_q_->forward(query);
    if (norm_k_) key = norm_k_->forward(key);
    value = all_to_all_4D_post2(handle_v);
    encoder_hidden_states_query_proj = all_to_all_4D_post2(handle_eq);
    encoder_hidden_states_key_proj = all_to_all_4D_post2(handle_ek);
    if (norm_added_q_)
      encoder_hidden_states_query_proj =
          norm_added_q_->forward(encoder_hidden_states_query_proj);
    if (norm_added_k_)
      encoder_hidden_states_key_proj =
          norm_added_k_->forward(encoder_hidden_states_key_proj);
    encoder_hidden_states_value_proj = all_to_all_4D_post2(handle_ev);
    query =
        torch::cat({encoder_hidden_states_query_proj, query}, 1).contiguous();
    key = torch::cat({encoder_hidden_states_key_proj, key}, 1).contiguous();
    value =
        torch::cat({encoder_hidden_states_value_proj, value}, 1).contiguous();
    return std::make_tuple(query,
                           key,
                           value,
                           encoder_hidden_states_query_proj,
                           encoder_hidden_states_key_proj,
                           encoder_hidden_states_value_proj);
  }

  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  qkv_matmul(const torch::Tensor& hidden_states_reshaped,
             const torch::Tensor& encoder_hidden_states_reshaped,
             int64_t attn_heads,
             int64_t head_dim) {
    int64_t batch_size = encoder_hidden_states_reshaped.size(0);
    torch::Tensor query = to_q_->forward(hidden_states_reshaped);
    torch::Tensor key = to_k_->forward(hidden_states_reshaped);
    torch::Tensor value = to_v_->forward(hidden_states_reshaped);
    query = query.view({batch_size, -1, attn_heads, head_dim});  // BSND
    key = key.view({batch_size, -1, attn_heads, head_dim});      // BSND
    value = value.view({batch_size, -1, attn_heads, head_dim});  // BSND
    if (norm_q_) query = norm_q_->forward(query);
    if (norm_k_) key = norm_k_->forward(key);
    torch::Tensor encoder_hidden_states_query_proj =
        add_q_proj_->forward(encoder_hidden_states_reshaped);
    torch::Tensor encoder_hidden_states_key_proj =
        add_k_proj_->forward(encoder_hidden_states_reshaped);
    torch::Tensor encoder_hidden_states_value_proj =
        add_v_proj_->forward(encoder_hidden_states_reshaped);
    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        {batch_size, -1, attn_heads, head_dim});
    if (norm_added_q_)
      encoder_hidden_states_query_proj =
          norm_added_q_->forward(encoder_hidden_states_query_proj);

    if (norm_added_k_)
      encoder_hidden_states_key_proj =
          norm_added_k_->forward(encoder_hidden_states_key_proj);
    // TODO some are right some are wrong query1& key1.
    // encoder_hidden_states_query_proj
    auto query1 =
        torch::cat({encoder_hidden_states_query_proj, query}, 1).contiguous();
    auto key1 =
        torch::cat({encoder_hidden_states_key_proj, key}, 1).contiguous();
    auto value1 =
        torch::cat({encoder_hidden_states_value_proj, value}, 1).contiguous();
    return std::make_tuple(query1,
                           key1,
                           value1,
                           encoder_hidden_states_query_proj,
                           encoder_hidden_states_key_proj,
                           encoder_hidden_states_value_proj);
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& image_rotary_emb) {
    int64_t input_ndim = hidden_states.dim();
    torch::Tensor hidden_states_reshaped = hidden_states;
    if (input_ndim == 4) {
      auto shape = hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      hidden_states_reshaped =
          hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    int64_t context_input_ndim = encoder_hidden_states.dim();
    torch::Tensor encoder_hidden_states_reshaped = encoder_hidden_states;
    if (context_input_ndim == 4) {
      auto shape = encoder_hidden_states.sizes();
      int64_t batch_size = shape[0];
      int64_t channel = shape[1];
      int64_t height = shape[2];
      int64_t width = shape[3];
      encoder_hidden_states_reshaped =
          encoder_hidden_states.view({batch_size, channel, height * width})
              .transpose(1, 2);
    }
    int64_t batch_size = encoder_hidden_states_reshaped.size(0);
    int64_t inner_dim = to_k_->weight_.size(0);
    int64_t attn_heads = heads_;
    int64_t head_dim = inner_dim / attn_heads;
    torch::Tensor query1, key1, value1;
    torch::Tensor encoder_hidden_states_query_proj,
        encoder_hidden_states_key_proj, encoder_hidden_states_value_proj;
    // Compute QKV projections
    if (use_sp_) {
      std::tie(query1,
               key1,
               value1,
               encoder_hidden_states_query_proj,
               encoder_hidden_states_key_proj,
               encoder_hidden_states_value_proj) =
          cmo_matmul_all2all(hidden_states_reshaped,
                             encoder_hidden_states_reshaped,
                             attn_heads,
                             head_dim);

    } else {
      std::tie(query1,
               key1,
               value1,
               encoder_hidden_states_query_proj,
               encoder_hidden_states_key_proj,
               encoder_hidden_states_value_proj) =
          qkv_matmul(hidden_states_reshaped,
                     encoder_hidden_states_reshaped,
                     attn_heads,
                     head_dim);
    }
    if (image_rotary_emb.defined()) {
      query1 = apply_rotary_emb(query1, image_rotary_emb, false);
      key1 = apply_rotary_emb(key1, image_rotary_emb, false);
    }
    // CHECK_EQ(false,true);
    //  torch::Tensor attn_output = torch::scaled_dot_product_attention(
    //      query1, key1, value1, torch::nullopt, 0.0, false);
    int64_t head_num_ = query1.size(2);
    int64_t head_dim_ = query1.size(-1);
    auto results =
        at_npu::native::custom_ops::npu_fusion_attention(query1,
                                                         key1,
                                                         value1,
                                                         head_num_,
                                                         "BSND",
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         torch::nullopt,
                                                         pow(head_dim_, -0.5),
                                                         1.0,
                                                         65535,
                                                         65535);
    auto attn_output = std::get<0>(results);
    attn_output = attn_output.reshape({batch_size, -1, head_dim_ * head_num_});
    attn_output = attn_output.to(query1.dtype());
    int64_t encoder_length = encoder_hidden_states_query_proj.size(1);
    torch::Tensor encoder_output = attn_output.slice(1, 0, encoder_length);
    torch::Tensor hidden_output = attn_output.slice(1, encoder_length);
    encoder_output = encoder_output.flatten(2);
    hidden_output = hidden_output.flatten(2);  // (B, L, H*D)
    attn_heads = heads_ / world_size_;
    AllToAll4DHandle handle_e, handle_h;
    if (use_sp_) {
      hidden_output =
          hidden_output.view({batch_size, -1, attn_heads, head_dim});  // BSND
      handle_h =
          all_to_all_4D(hidden_output, rank_, world_size_, 1, 2, true, pg_);
      hidden_output = all_to_all_4D_post(handle_h);
      hidden_output = hidden_output.view({batch_size, -1, inner_dim});
      encoder_output =
          encoder_output.view({batch_size, -1, attn_heads, head_dim});
      handle_e =
          all_to_all_4D(encoder_output, rank_, world_size_, 1, 2, false, pg_);
      // hidden_output = all_to_all_4D_post(handle_h);
      // hidden_output = hidden_output.view({batch_size, -1, inner_dim});
    }
    hidden_output = to_out_->forward(hidden_output);
    // both are wrong
    if (use_sp_) {
      encoder_output = all_to_all_4D_post(handle_e);
      encoder_output = encoder_output.view({batch_size, -1, inner_dim});
    }
    encoder_output = to_add_out_->forward(encoder_output);
    return std::make_tuple(hidden_output, encoder_output);
  }
};
TORCH_MODULE(FluxAttention);

class PixArtAlphaTextProjectionImpl : public torch::nn::Module {
 public:
  PixArtAlphaTextProjectionImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    int64_t hidden_size = model_args.dit_attention_head_dim() *
                          model_args.dit_num_attention_heads();
    int64_t in_features = model_args.dit_pooled_projection_dim();
    int64_t out_dim =
        hidden_size;  //(out_features == -1) ? hidden_size : out_features;
    linear_1_ = register_module(
        "linear_1",
        xllm_ops::DiTLinear(in_features, hidden_size, true, options_));

    linear_2_ = register_module(
        "linear_2", xllm_ops::DiTLinear(hidden_size, out_dim, true, options_));
    act_1_ = register_module("act_1", torch::nn::SiLU());
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear_1
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    // linear_2
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  torch::Tensor forward(const torch::Tensor& caption) {
    auto hidden_states = linear_1_->forward(caption);
    hidden_states = act_1_->forward(hidden_states);
    hidden_states = linear_2_->forward(hidden_states);
    return hidden_states;
  }

 private:
  xllm_ops::DiTLinear linear_1_{nullptr};
  xllm_ops::DiTLinear linear_2_{nullptr};
  torch::nn::SiLU act_1_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(PixArtAlphaTextProjection);

inline torch::Tensor get_timestep_embedding(const torch::Tensor& timesteps,
                                            int64_t embedding_dim,
                                            bool flip_sin_to_cos = false,
                                            float downscale_freq_shift = 1.0f,
                                            float scale = 1.0f,
                                            int64_t max_period = 10000) {
  TORCH_CHECK(timesteps.dim() == 1, "Timesteps should be a 1d-array");
  int64_t half_dim = embedding_dim / 2;
  // -ln(max_period) * [0, 1, ..., half_dim-1] / (half_dim -
  // downscale_freq_shift
  torch::TensorOptions options = timesteps.options();
  auto exponent = -std::log(static_cast<float>(max_period)) *
                  torch::arange(/*start=*/0,
                                /*end=*/half_dim,
                                /*step=*/1,
                                options);
  exponent = exponent / (half_dim - downscale_freq_shift);

  // timesteps[:, None] * exp(exponent)[None, :]
  auto emb = torch::exp(exponent);                  // [half_dim]
  emb = timesteps.unsqueeze(1) * emb.unsqueeze(0);  // [N, half_dim]
  emb = scale * emb;

  // [sin(emb), cos(emb)]
  auto sin_emb = torch::sin(emb);
  auto cos_emb = torch::cos(emb);
  auto combined =
      torch::cat({sin_emb, cos_emb}, /*dim=*/-1);  // [N, 2*half_dim]

  if (flip_sin_to_cos) {
    combined = torch::cat(
        {combined.slice(
             /*dim=*/-1, /*start=*/half_dim, /*end=*/2 * half_dim),   // cos
         combined.slice(/*dim=*/-1, /*start=*/0, /*end=*/half_dim)},  // sin
        /*dim=*/-1);
  }

  if (embedding_dim % 2 == 1) {
    combined = torch::nn::functional::pad(
        combined, torch::nn::functional::PadFuncOptions({0, 1, 0, 0}));
  }

  return combined;  // [N, embedding_dim]
}

class TimestepsImpl : public torch::nn::Module {
 public:
  TimestepsImpl(int64_t num_channels,
                bool flip_sin_to_cos,
                float downscale_freq_shift,
                int64_t scale = 1)
      : num_channels_(num_channels),
        flip_sin_to_cos_(flip_sin_to_cos),
        downscale_freq_shift_(downscale_freq_shift),
        scale_(scale) {}

  torch::Tensor forward(const torch::Tensor& timesteps) {
    return get_timestep_embedding(timesteps,
                                  num_channels_,
                                  flip_sin_to_cos_,
                                  downscale_freq_shift_,
                                  scale_,
                                  10000  // max_period
    );
  }

 private:
  int64_t num_channels_;
  bool flip_sin_to_cos_;
  float downscale_freq_shift_;
  int64_t scale_;
};
TORCH_MODULE(Timesteps);

class TimestepEmbeddingImpl : public torch::nn::Module {
 public:
  TimestepEmbeddingImpl(int64_t in_channels,
                        int64_t time_embed_dim,
                        int64_t out_dim,
                        int64_t cond_proj_dim,
                        bool sample_proj_bias,
                        torch::TensorOptions& options)
      : options_(options) {
    linear_1_ = register_module(
        "linear_1",
        xllm_ops::DiTLinear(
            in_channels, time_embed_dim, sample_proj_bias, options_));

    act_fn_ = register_module("act_fn", torch::nn::SiLU());

    int64_t time_embed_dim_out = (out_dim == -1) ? time_embed_dim : out_dim;
    linear_2_ = register_module(
        "linear_2",
        xllm_ops::DiTLinear(
            time_embed_dim, time_embed_dim_out, sample_proj_bias, options_));
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear1
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    // linear2
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }

  torch::Tensor forward(const torch::Tensor& sample,
                        const torch::Tensor& condition = torch::Tensor()) {
    torch::Tensor x1 = linear_1_->forward(sample);
    x1 = act_fn_->forward(x1);
    x1 = linear_2_->forward(x1);
    return x1;
  }

 private:
  xllm_ops::DiTLinear linear_1_{nullptr};
  xllm_ops::DiTLinear linear_2_{nullptr};
  xllm_ops::DiTLinear cond_proj_{nullptr};
  torch::nn::SiLU post_act_{nullptr};
  torch::nn::SiLU act_fn_{nullptr};
  // bool has_cond_proj_;
  torch::TensorOptions options_;
};
TORCH_MODULE(TimestepEmbedding);

class LabelEmbeddingImpl : public torch::nn::Module {
 public:
  LabelEmbeddingImpl(int64_t num_classes,
                     int64_t hidden_size,
                     float dropout_prob)
      : num_classes_(num_classes), dropout_prob_(dropout_prob) {
    bool use_cfg_embedding = dropout_prob > 0;
    embedding_table_ = register_module(
        "embedding_table",
        torch::nn::Embedding(num_classes + use_cfg_embedding, hidden_size));
  }

  torch::Tensor token_drop(
      torch::Tensor labels,
      c10::optional<torch::Tensor> force_drop_ids = c10::nullopt) {
    torch::Tensor drop_ids;
    if (!force_drop_ids.has_value()) {
      drop_ids = torch::rand({labels.size(0)}, labels.device()) < dropout_prob_;
    } else {
      drop_ids = force_drop_ids.value() == 1;
    }

    torch::Tensor mask = torch::full_like(labels, num_classes_);
    labels = torch::where(drop_ids, mask, labels);
    return labels;
  }

  torch::Tensor forward(
      torch::Tensor labels,
      c10::optional<torch::Tensor> force_drop_ids = c10::nullopt) {
    bool use_dropout = dropout_prob_ > 0;
    if ((is_training() && use_dropout) || force_drop_ids.has_value()) {
      labels = token_drop(labels, force_drop_ids);
    }

    torch::Tensor embeddings = embedding_table_->forward(labels);
    return embeddings;
  }

 private:
  torch::nn::Embedding embedding_table_{nullptr};
  int64_t num_classes_;
  float dropout_prob_;
};
TORCH_MODULE(LabelEmbedding);

class CombinedTimestepTextProjEmbeddingsImpl : public torch::nn::Module {
 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
  torch::TensorOptions options_;

 public:
  CombinedTimestepTextProjEmbeddingsImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto embedding_dim = model_args.dit_attention_head_dim() *
                         model_args.dit_num_attention_heads();
    auto pooled_projection_dim = model_args.dit_pooled_projection_dim();
    // num_channels=256, flip_sin_to_cos=true,
    // downscale_freq_shift=0, scale=1
    time_proj_ = Timesteps(256, true, 0.0f, 1);

    timestep_embedder_ = TimestepEmbedding(
        256,
        embedding_dim,
        -1,
        -1,
        true,
        options_);  // in_channels=256, time_embed_dim=embedding_dim
    text_embedder_ = PixArtAlphaTextProjection(context);
  }

  void load_state_dict(const StateDict& state_dict) {
    // timestep_embedder
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }

  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_(timestep);
    auto timesteps_emb = timestep_embedder_(timesteps_proj);

    auto pooled_projections = text_embedder_(pooled_projection);
    return timesteps_emb + pooled_projections;
  }
};
TORCH_MODULE(CombinedTimestepTextProjEmbeddings);

class CombinedTimestepGuidanceTextProjEmbeddingsImpl
    : public torch::nn::Module {
 private:
  TimestepEmbedding guidance_embedder_{nullptr};
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  PixArtAlphaTextProjection text_embedder_{nullptr};
  torch::TensorOptions options_;

 public:
  void load_state_dict(const StateDict& state_dict) {
    // guidance_embedder
    guidance_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("guidance_embedder."));
    // timestep_embedder
    timestep_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("timestep_embedder."));
    // text_embedder
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }
  CombinedTimestepGuidanceTextProjEmbeddingsImpl(const ModelContext& context)
      : options_(context.get_tensor_options()),
        time_proj_(256,
                   true,
                   0.0f,
                   1)  // num_channels=256, flip_sin_to_cos=true,
                       // downscale_freq_shift=0, scale=1
  {
    auto model_args = context.get_model_args();
    auto embedding_dim = model_args.dit_attention_head_dim() *
                         model_args.dit_num_attention_heads();
    auto pooled_projection_dim = model_args.dit_pooled_projection_dim();

    text_embedder_ = PixArtAlphaTextProjection(context);
    timestep_embedder_ = TimestepEmbedding(
        256,
        embedding_dim,
        -1,
        -1,
        true,
        options_);  // in_channels=256, time_embed_dim=embedding_dim
    guidance_embedder_ = TimestepEmbedding(
        256, embedding_dim, -1, -1, true, options_);  // in_channels=256
  }
  torch::Tensor forward(const torch::Tensor& timestep,
                        const torch::Tensor& guidance,
                        const torch::Tensor& pooled_projection) {
    auto timesteps_proj = time_proj_->forward(timestep);  // [N, 256]
    auto timesteps_emb =
        timestep_embedder_->forward(timesteps_proj);     // [N, embedding_dim]
    auto guidance_proj = time_proj_->forward(guidance);  // [N, 256]
    auto guidance_emb =
        guidance_embedder_->forward(guidance_proj);  // [N, embedding_dim]
    auto time_guidance_emb =
        timesteps_emb + guidance_emb;  // [N, embedding_dim]
    auto pooled_projections =
        text_embedder_->forward(pooled_projection);  // [N, embedding_dim]
    return time_guidance_emb + pooled_projections;   // [N, embedding_dim]
  }
};
TORCH_MODULE(CombinedTimestepGuidanceTextProjEmbeddings);

class CombinedTimestepLabelEmbeddingsImpl : public torch::nn::Module {
 public:
  CombinedTimestepLabelEmbeddingsImpl(int64_t num_classes,
                                      int64_t embedding_dim,
                                      float class_dropout_prob,
                                      torch::TensorOptions& options) {
    time_proj_ = register_module("time_proj", Timesteps(256, true, 1, 1));
    timestep_embedder_ = register_module(
        "timestep_embedder",
        TimestepEmbedding(256, embedding_dim, -1, -1, true, options));
    class_embedder_ = register_module(
        "class_embedder",
        LabelEmbedding(num_classes, embedding_dim, class_dropout_prob));
  }

  torch::Tensor forward(torch::Tensor timestep, torch::Tensor class_labels) {
    torch::Tensor timesteps_proj = time_proj_(timestep);

    torch::Tensor timesteps_emb;

    timesteps_emb = timestep_embedder_(timesteps_proj);

    torch::Tensor class_emb = class_embedder_(class_labels);

    torch::Tensor conditioning = timesteps_emb + class_emb;

    return conditioning;
  }

 private:
  Timesteps time_proj_{nullptr};
  TimestepEmbedding timestep_embedder_{nullptr};
  LabelEmbedding class_embedder_{nullptr};
};
TORCH_MODULE(CombinedTimestepLabelEmbeddings);

class AdaLayerNormZeroImpl : public torch::nn::Module {
 public:
  AdaLayerNormZeroImpl(int64_t embedding_dim,
                       int64_t num_embeddings,
                       bool bias,
                       torch::TensorOptions options)
      : options_(options) {
    if (num_embeddings > 0) {
      emb_ = register_module("emb",
                             CombinedTimestepLabelEmbeddings(
                                 num_embeddings, embedding_dim, 0.1, options_));
    }
    silu_ = register_module("silu", torch::nn::SiLU());

    linear_ = register_module(
        "linear",
        xllm_ops::DiTLinear(embedding_dim, 6 * embedding_dim, bias, options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));
  }
  std::tuple<torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor,
             torch::Tensor>
  forward(const torch::Tensor& x,
          const torch::Tensor& timestep = torch::Tensor(),
          const torch::Tensor& class_labels = torch::Tensor(),
          const torch::Tensor& emb = torch::Tensor()) {
    torch::Tensor ada_emb = emb;
    if (!emb_.is_empty()) {
      ada_emb = emb_->forward(timestep, class_labels);
    }
    ada_emb = linear_->forward(silu_->forward(ada_emb));
    auto splits = torch::chunk(ada_emb, 6, 1);

    auto shift_msa = splits[0];
    auto scale_msa = splits[1];
    auto gate_msa = splits[2];
    auto shift_mlp = splits[3];
    auto scale_mlp = splits[4];
    auto gate_mlp = splits[5];

    auto normalized_x = norm_->forward(x) * (1 + scale_msa.unsqueeze(1)) +
                        shift_msa.unsqueeze(1);
    return {normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp};
  }

  void load_state_dict(const StateDict& state_dict) {
    // linear_->to(device_);
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  xllm_ops::DiTLinear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  CombinedTimestepLabelEmbeddings emb_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormZero);

class AdaLayerNormZeroSingleImpl : public torch::nn::Module {
 public:
  AdaLayerNormZeroSingleImpl(int64_t embedding_dim,
                             bool bias,
                             torch::TensorOptions& options)
      : options_(options) {
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear",
        xllm_ops::DiTLinear(embedding_dim, 3 * embedding_dim, bias, options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(false)
                                 .eps(1e-6)));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& emb = torch::Tensor()) {
    auto ada_emb = linear_->forward(silu_->forward(emb));
    auto splits = torch::chunk(ada_emb, 3, 1);

    auto shift_msa = splits[0];
    auto scale_msa = splits[1];
    auto gate_msa = splits[2];

    auto normalized_x = norm_->forward(x) * (1 + scale_msa.unsqueeze(1)) +
                        shift_msa.unsqueeze(1);

    return {normalized_x, gate_msa};
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  xllm_ops::DiTLinear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormZeroSingle);

class AdaLayerNormContinuousImpl : public torch::nn::Module {
 public:
  AdaLayerNormContinuousImpl(int64_t embedding_dim,
                             int64_t conditioning_embedding_dim,
                             bool elementwise_affine,
                             double eps,
                             bool bias,
                             torch::TensorOptions& options)
      : options_(options) {
    silu_ = register_module("silu", torch::nn::SiLU());
    linear_ = register_module(
        "linear",
        xllm_ops::DiTLinear(
            conditioning_embedding_dim, 2 * embedding_dim, bias, options_));
    norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})
                                 .elementwise_affine(elementwise_affine)
                                 .eps(eps)));
  }

  torch::Tensor forward(const torch::Tensor& x,
                        const torch::Tensor& conditioning_embedding) {
    auto cond_emb = silu_->forward(conditioning_embedding);
    cond_emb = cond_emb.to(x.dtype());

    auto emb = linear_->forward(cond_emb);
    auto chunks = torch::chunk(emb, 2, 1);
    torch::Tensor scale, shift;

    scale = chunks[0];
    shift = chunks[1];
    auto x_norm = norm_->forward(x);
    return x_norm * (1 + scale).unsqueeze(1) + shift.unsqueeze(1);
  }

  void load_state_dict(const StateDict& state_dict) {
    //  linear
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

 private:
  torch::nn::SiLU silu_{nullptr};
  xllm_ops::DiTLinear linear_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::string norm_type_;
  double eps_;
  bool elementwise_affine_;
  torch::Tensor rms_scale_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(AdaLayerNormContinuous);

class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();
    auto dim = num_attention_heads * attention_head_dim;
    auto inner_dim = dim * 4;
    auto dim_out = dim;

    // linear1
    linear1_ = register_module(
        "linear1", xllm_ops::DiTLinear(dim, inner_dim, true, options_));
    // activation
    activation_ = register_module(
        "activation",
        torch::nn::Functional(std::function<at::Tensor(const at::Tensor&)>(
            [](const at::Tensor& x) { return torch::gelu(x, "tanh"); })));

    // linear2

    linear2_ = register_module(
        "linear2", xllm_ops::DiTLinear(inner_dim, dim_out, true, options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor out = linear1_->forward(hidden_states);
    out = activation_(out);
    out = linear2_->forward(out);
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear1_->load_state_dict(state_dict.get_dict_with_prefix("net.0.proj."));
    linear2_->load_state_dict(state_dict.get_dict_with_prefix("net.2."));
  }

 private:
  xllm_ops::DiTLinear linear1_{nullptr};
  torch::nn::Functional activation_{nullptr};
  xllm_ops::DiTLinear linear2_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(FeedForward);

class FluxSingleTransformerBlockImpl : public torch::nn::Module {
 public:
  FluxSingleTransformerBlockImpl(const ModelContext& context, int layer_id)
      : options_(context.get_tensor_options()),
        layer_id_(layer_id),
        num_layers_(context.get_model_args().dit_num_single_layers()),
        pg_(context.get_parallel_args()) {
    auto model_args = context.get_model_args();
    world_size_ = pg_.world_size();
    use_sp_ = world_size_ > 1;
    rank_ = pg_.rank();
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();
    auto dim = num_attention_heads * attention_head_dim;
    mlp_hidden_dim_ = dim * 4;

    norm_ =
        register_module("norm", AdaLayerNormZeroSingle(dim, true, options_));

    int64_t mlp_out_dim = mlp_hidden_dim_;
    proj_mlp_ = register_module(
        "proj_mlp", xllm_ops::DiTLinear(dim, mlp_out_dim, true, options_));

    int64_t proj_in_dim = dim + mlp_hidden_dim_;
    int64_t proj_out_dim = dim;
    proj_out_ = register_module(
        "proj_out",
        xllm_ops::DiTLinear(proj_in_dim, proj_out_dim, true, options_));
    act_mlp_ =
        register_module("gelu",
                        torch::nn::Functional(
                            std::function<torch::Tensor(const torch::Tensor&)>(
                                [](const torch::Tensor& x) {
                                  return torch::gelu(x, "tanh");
                                })));

    attn_ = register_module("attn", FluxSingleAttention(context));
  }
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    torch::Tensor hidden_states_ = hidden_states;
    if (use_sp_ && layer_id_ == 0) {
      int32_t seq_len = hidden_states_.size(1);
      int64_t pad = (world_size_ - (seq_len % world_size_)) % world_size_;
      hidden_states_ =
          split_sequence(hidden_states_, world_size_, rank_, 1, pad);
    }
    auto residual = hidden_states_;
    auto [norm_hidden_states, gate] = norm_(hidden_states_, temb);
    auto mlp_hidden_states = act_mlp_(proj_mlp_(norm_hidden_states));
    auto attn_output = attn_->forward(norm_hidden_states, image_rotary_emb);
    auto hidden_states_cat = torch::cat({attn_output, mlp_hidden_states}, 2);
    auto out = proj_out_(hidden_states_cat);
    out = gate.unsqueeze(1) * out;
    out = residual + out;
    if (use_sp_ && layer_id_ == num_layers_ - 1) {
      int32_t seq_len = out.size(1);
      int64_t pad = (world_size_ - (seq_len % world_size_)) % world_size_;
      out = gather_sequence(out, world_size_, 1, pad, pg_);
    }
    // if (out.scalar_type() == torch::kFloat16) {
    //   out = torch::clamp(out, -65504.0f, 65504.0f);
    // }
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    // attn
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    // norm
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    // proj_mlp

    proj_mlp_->load_state_dict(state_dict.get_dict_with_prefix("proj_mlp."));
    // proj_out_
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix("proj_out."));
  }

 private:
  AdaLayerNormZeroSingle norm_{nullptr};
  xllm_ops::DiTLinear proj_mlp_{nullptr};
  xllm_ops::DiTLinear proj_out_{nullptr};
  torch::nn::Functional act_mlp_{nullptr};
  FluxSingleAttention attn_{nullptr};
  int64_t mlp_hidden_dim_;
  torch::TensorOptions options_;

  // For sequence parallel
  int layer_id_;
  int num_layers_;
  int world_size_{1};
  int rank_{0};
  bool use_sp_{false};
  ParallelArgs pg_;
};
TORCH_MODULE(FluxSingleTransformerBlock);

class FluxTransformerBlockImpl : public torch::nn::Module {
 public:
  FluxTransformerBlockImpl(const ModelContext& context, int layer_id)
      : options_(context.get_tensor_options()),
        layer_id_(layer_id),
        num_layers_(context.get_model_args().dit_num_layers()),
        pg_(context.get_parallel_args()) {
    auto model_args = context.get_model_args();
    use_sp_ = pg_.world_size() > 1;
    world_size_ = pg_.world_size();
    rank_ = pg_.rank();
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();

    auto dim = num_attention_heads * attention_head_dim;
    double eps = 1e-6;

    norm1_ = register_module("norm1",
                             AdaLayerNormZero(dim, 0, true /*bias*/, options_));

    norm1_context_ = register_module(
        "norm1_context", AdaLayerNormZero(dim, 0, true /*bias*/, options_));

    attn_ = register_module("attn", FluxAttention(context));
    norm2_ = register_module(
        "norm2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_ = register_module("ff", FeedForward(context));
    norm2_context_ = register_module(
        "norm2_context",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({dim}).elementwise_affine(false).eps(
                eps)));

    ff_context_ = register_module("ff_context", FeedForward(context));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb,
      const torch::Tensor& image_rotary_emb = torch::Tensor()) {
    torch::Tensor hidden_states_ = hidden_states;
    torch::Tensor encoder_hidden_states_ = encoder_hidden_states;
    int32_t seq_len, encoder_seq_len, pad, encoder_pad;
    // split the sequence for hidden_states and the encoder_hidden_states
    if (use_sp_ && layer_id_ == 0) {
      seq_len = hidden_states_.size(1);
      encoder_seq_len = encoder_hidden_states_.size(1);
      pad = (world_size_ - (seq_len % world_size_)) % world_size_;
      encoder_pad =
          (world_size_ - (encoder_seq_len % world_size_)) % world_size_;
      hidden_states_ =
          split_sequence(hidden_states_, world_size_, rank_, 1, pad);
      encoder_hidden_states_ = split_sequence(
          encoder_hidden_states_, world_size_, rank_, 1, encoder_pad);
    }

    auto [norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp] =
        norm1_(hidden_states_, torch::Tensor(), torch::Tensor(), temb);
    auto [norm_encoder_hidden_states,
          c_gate_msa,
          c_shift_mlp,
          c_scale_mlp,
          c_gate_mlp] =
        norm1_context_(
            encoder_hidden_states_, torch::Tensor(), torch::Tensor(), temb);
    auto [attn_output, context_attn_output] =
        attn_(norm_hidden_states, norm_encoder_hidden_states, image_rotary_emb);
    attn_output = gate_msa.unsqueeze(1) * attn_output;
    auto new_hidden_states = hidden_states_ + attn_output;
    // image latent
    auto norm_hs = norm2_(new_hidden_states);
    norm_hs = norm_hs * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1);
    auto ff_output = ff_->forward(norm_hs);
    new_hidden_states = new_hidden_states + gate_mlp.unsqueeze(1) * ff_output;
    // context
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output;
    auto new_encoder_hidden_states =
        encoder_hidden_states_ + context_attn_output;
    auto norm_enc_hs = norm2_context_(new_encoder_hidden_states);
    norm_enc_hs =
        norm_enc_hs * (1 + c_scale_mlp.unsqueeze(1)) + c_shift_mlp.unsqueeze(1);
    auto ff_context_out = ff_context_->forward(norm_enc_hs);
    new_encoder_hidden_states =
        new_encoder_hidden_states + c_gate_mlp.unsqueeze(1) * ff_context_out;
    if (new_encoder_hidden_states.scalar_type() == torch::kFloat16) {
      new_encoder_hidden_states =
          torch::clamp(new_encoder_hidden_states, -65504.0f, 65504.0f);
    }
    // gather the full sequence for hidden_states and the
    // encoder_hidden_states
    if (use_sp_ && (layer_id_ == num_layers_ - 1)) {
      pad = (world_size_ - (seq_len % world_size_)) % world_size_;
      encoder_pad =
          (world_size_ - (encoder_seq_len % world_size_)) % world_size_;
      new_hidden_states =
          gather_sequence(new_hidden_states, world_size_, 1, pad, pg_);
      new_encoder_hidden_states = gather_sequence(
          new_encoder_hidden_states, world_size_, 1, encoder_pad, pg_);
    }
    return std::make_tuple(new_hidden_states, new_encoder_hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm1
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    // norm1_context
    norm1_context_->load_state_dict(
        state_dict.get_dict_with_prefix("norm1_context."));
    // attn
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    // ff
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    // ff_context
    ff_context_->load_state_dict(
        state_dict.get_dict_with_prefix("ff_context."));
  }

 private:
  AdaLayerNormZero norm1_{nullptr};
  AdaLayerNormZero norm1_context_{nullptr};
  FluxAttention attn_{nullptr};
  torch::nn::LayerNorm norm2_{nullptr};
  FeedForward ff_{nullptr};
  torch::nn::LayerNorm norm2_context_{nullptr};
  FeedForward ff_context_{nullptr};
  torch::TensorOptions options_;

  // sequence parallel flag
  int layer_id_;
  int num_layers_;
  int world_size_{1};
  int rank_{0};
  bool use_sp_{false};
  ParallelArgs pg_;
};
TORCH_MODULE(FluxTransformerBlock);

class FluxTransformer2DModelImpl : public torch::nn::Module {
 public:
  FluxTransformer2DModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()),
        pg_(context.get_parallel_args()) {
    auto model_args = context.get_model_args();
    use_sp_ = context.get_parallel_args().world_size() > 1;
    auto num_attention_heads = model_args.dit_num_attention_heads();
    auto attention_head_dim = model_args.dit_attention_head_dim();
    auto inner_dim = num_attention_heads * attention_head_dim;
    auto pooled_projection_dim = model_args.dit_pooled_projection_dim();
    auto joint_attention_dim = model_args.dit_joint_attention_dim();
    auto axes_dims_rope = model_args.dit_axes_dims_rope();
    auto num_layers = model_args.dit_num_layers();
    auto num_single_layers = model_args.dit_num_single_layers();
    auto patch_size = model_args.dit_patch_size();
    out_channels_ = model_args.dit_in_channels();
    guidance_embeds_ = model_args.dit_guidance_embeds();

    // Initialize the transformer model components here
    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    single_transformer_blocks_ =
        register_module("single_transformer_blocks", torch::nn::ModuleList());
    if (guidance_embeds_) {
      time_text_guidance_embed_ =
          register_module("time_text_guidance_embed",
                          CombinedTimestepGuidanceTextProjEmbeddings(context));
    } else {
      time_text_embed_ = register_module(
          "time_text_embed", CombinedTimestepTextProjEmbeddings(context));
    }
    context_embedder_ = register_module(
        "context_embedder",
        xllm_ops::DiTLinear(joint_attention_dim, inner_dim, true, options_));
    x_embedder_ = register_module(
        "x_embedder",
        xllm_ops::DiTLinear(out_channels_, inner_dim, true, options_));
    // mm-dit block
    for (int64_t i = 0; i < num_layers; ++i) {
      transformer_blocks_->push_back(FluxTransformerBlock(context, i));
    }
    // single mm-dit block
    for (int64_t i = 0; i < num_single_layers; ++i) {
      single_transformer_blocks_->push_back(
          FluxSingleTransformerBlock(context, i));
    }
    norm_out_ =
        register_module("norm_out",
                        AdaLayerNormContinuous(inner_dim,
                                               inner_dim,
                                               false, /*elementwise_affine*/
                                               1e-6,  /*eps*/
                                               true,  /*eps*/
                                               options_));
    proj_out_ = register_module(
        "proj_out",
        xllm_ops::DiTLinear(inner_dim,
                            patch_size * patch_size * out_channels_,
                            true,
                            options_));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor hidden_states = x_embedder_->forward(hidden_states_input);
    auto timestep_scaled = timestep.to(hidden_states.dtype()) * 1000.0f;
    torch::Tensor temb;
    if (guidance.defined()) {
      auto guidance_scaled = guidance.to(hidden_states.dtype()) * 1000.0f;
      temb = time_text_guidance_embed_->forward(
          timestep_scaled, guidance_scaled, pooled_projections);
    } else {
      temb = time_text_embed_->forward(timestep_scaled, pooled_projections);
    }
    torch::Tensor encoder_hidden_states =
        context_embedder_->forward(encoder_hidden_states_input);

    bool use_step_cache = false;
    bool use_block_cache = false;
    torch::Tensor original_hidden_states = hidden_states;
    torch::Tensor original_encoder_hidden_states = encoder_hidden_states;
    // {
    //   // step_begin: input: hidden_states, original_hidden_states
    //   TensorMap step_in_map = {
    //       {"hidden_states", hidden_states},
    //       {"original_hidden_states", original_hidden_states}};
    //   CacheStepIn stepin_before(step_idx, step_in_map);
    //   use_step_cache = DiTCache::getinstance().on_before_step(stepin_before);
    // }
    if (!use_step_cache) {
      for (int64_t i = 0; i < transformer_blocks_->size(); ++i) {
        // {
        //   // transformer_block begin: input: block_id
        //   CacheBlockIn blockin_before(i);
        //   use_block_cache =
        //       DiTCache::getinstance().on_before_block(blockin_before);
        // }

        if (!use_block_cache) {
          auto block = transformer_blocks_[i]->as<FluxTransformerBlock>();
          auto [new_hidden, new_encoder_hidden] = block->forward(
              hidden_states, encoder_hidden_states, temb, image_rotary_emb);
          hidden_states = new_hidden;
          encoder_hidden_states = new_encoder_hidden;
        }
        // {
        //   // transformer_block after: input: block_id, hidden_states,
        //   // encoder_hidden_states, original_hidden_states,
        //   // original_encoder_hidden_states
        //   TensorMap block_in_map = {
        //       {"hidden_states", hidden_states},
        //       {"encoder_hidden_states", encoder_hidden_states},
        //       {"original_hidden_states", original_hidden_states},
        //       {"original_encoder_hidden_states",
        //        original_encoder_hidden_states}};
        //   CacheBlockIn blockin_after(i, block_in_map);
        //   CacheBlockOut blockout_after =
        //       DiTCache::getinstance().on_after_block(blockin_after);

        //   hidden_states = blockout_after.tensors.at("hidden_states");
        //   encoder_hidden_states =
        //       blockout_after.tensors.at("encoder_hidden_states");
        // }
      }
      // // 1. 打印 hidden_states（shape: [1,8100,3072]）
      // print_tensor_for_compare(hidden_states, "hidden_states", 1000, 5);

      // // 2. 打印 encoder_hidden_states（shape: [1,512,3072]）
      // print_tensor_for_compare(encoder_hidden_states,
      // "encoder_hidden_states", 1000, 5);
      hidden_states = torch::cat({encoder_hidden_states, hidden_states}, 1);
      // CHECK_EQ(false, true);
      for (int64_t i = 0; i < single_transformer_blocks_->size(); ++i) {
        // {
        //   CacheBlockIn blockin_before(i);
        //   use_block_cache =
        //       DiTCache::getinstance().on_before_block(blockin_before);
        // }

        if (!use_block_cache) {
          auto block =
              single_transformer_blocks_[i]->as<FluxSingleTransformerBlock>();
          hidden_states = block->forward(hidden_states, temb, image_rotary_emb);
        }
        // {
        //   // single transformer_block after
        //   TensorMap single_block_map = {
        //       {"hidden_states", hidden_states},
        //       {"original_hidden_states", original_hidden_states}};
        //   CacheBlockIn blockin_after(i, single_block_map);
        //   CacheBlockOut blockout_after =
        //       DiTCache::getinstance().on_after_block(blockin_after);

        //   hidden_states = blockout_after.tensors.at("hidden_states");
        // }
      }

      int64_t start = encoder_hidden_states.size(1);
      int64_t length = hidden_states.size(1) - start;
      auto output_hidden =
          hidden_states.narrow(1, start, std::max(length, int64_t(0)));
      hidden_states = output_hidden;
    }

    // {
    //   // step after: input : hidden_states , original_hidden_states
    //   TensorMap step_after_map = {
    //       {"hidden_states", hidden_states},
    //       {"original_hidden_states", original_hidden_states}};
    //   CacheStepIn stepin_after(step_idx, step_after_map);
    //   CacheStepOut stepout_after =
    //       DiTCache::getinstance().on_after_step(stepin_after);
    //   hidden_states = stepout_after.tensors.at("hidden_states");
    // }
    auto output_hidden = norm_out_(hidden_states, temb);
    return proj_out_(output_hidden);
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    // Load model parameters from the loader
    for (const auto& state_dict : loader->get_state_dicts()) {
      // context_embedder
      context_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("context_embedder."));
      // x_embedder
      x_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("x_embedder."));
      // time_text_embed
      if (time_text_embed_) {
        time_text_embed_->load_state_dict(
            state_dict->get_dict_with_prefix("time_text_embed."));
      } else {
        time_text_guidance_embed_->load_state_dict(
            state_dict->get_dict_with_prefix("time_text_embed."));
      }
      // transformer_blocks
      for (int64_t i = 0; i < transformer_blocks_->size(); ++i) {
        auto block = transformer_blocks_[i]->as<FluxTransformerBlock>();
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "transformer_blocks." + std::to_string(i) + "."));
      }
      // single_transformer_blocks
      for (int64_t i = 0; i < single_transformer_blocks_->size(); ++i) {
        auto block =
            single_transformer_blocks_[i]->as<FluxSingleTransformerBlock>();
        block->load_state_dict(state_dict->get_dict_with_prefix(
            "single_transformer_blocks." + std::to_string(i) + "."));
      }
      // norm_out
      norm_out_->load_state_dict(state_dict->get_dict_with_prefix("norm_out."));
      // proj_out
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));
    }
  }
  int64_t in_channels() { return out_channels_; }
  bool guidance_embeds() { return guidance_embeds_; }

 private:
  CombinedTimestepTextProjEmbeddings time_text_embed_{nullptr};
  CombinedTimestepGuidanceTextProjEmbeddings time_text_guidance_embed_{nullptr};
  xllm_ops::DiTLinear context_embedder_{nullptr};
  xllm_ops::DiTLinear x_embedder_{nullptr};
  torch::nn::ModuleList transformer_blocks_{nullptr};
  torch::nn::ModuleList single_transformer_blocks_{nullptr};
  AdaLayerNormContinuous norm_out_{nullptr};
  xllm_ops::DiTLinear proj_out_{nullptr};
  bool guidance_embeds_;
  int64_t out_channels_;
  torch::TensorOptions options_;
  bool use_sp_{false};
  ParallelArgs pg_;
};
TORCH_MODULE(FluxTransformer2DModel);

class FluxDiTModelImpl : public torch::nn::Module {
 public:
  FluxDiTModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()), model_context_(context) {
    flux_transformer_2d_model_ = register_module(
        "flux_transformer_2d_model", FluxTransformer2DModel(context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states_input,
                        const torch::Tensor& encoder_hidden_states_input,
                        const torch::Tensor& pooled_projections,
                        const torch::Tensor& timestep,
                        const torch::Tensor& image_rotary_emb,
                        const torch::Tensor& guidance,
                        int64_t step_idx = 0) {
    torch::Tensor output =
        flux_transformer_2d_model_->forward(hidden_states_input,
                                            encoder_hidden_states_input,
                                            pooled_projections,
                                            timestep,
                                            image_rotary_emb,
                                            guidance,
                                            step_idx);
    return output;
  }
  int64_t in_channels() { return flux_transformer_2d_model_->in_channels(); }
  bool guidance_embeds() {
    return flux_transformer_2d_model_->guidance_embeds();
  }

  torch::Tensor _prepare_latent_image_ids(int64_t batch_size,
                                          int64_t height,
                                          int64_t width) {
    torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kInt64).device(options_.device());
    torch::Tensor latent_image_ids = torch::zeros({height, width, 3}, options);
    torch::Tensor row_indices = torch::arange(height, options).unsqueeze(1);
    latent_image_ids.select(2, 1) = row_indices;
    torch::Tensor col_indices = torch::arange(width, options).unsqueeze(0);
    latent_image_ids.select(2, 2) = col_indices;
    latent_image_ids = latent_image_ids.reshape({height * width, 3});

    return latent_image_ids;
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    flux_transformer_2d_model_->load_model(std::move(loader));
  }

 private:
  FluxTransformer2DModel flux_transformer_2d_model_{nullptr};
  torch::TensorOptions options_;
  ModelContext model_context_;
};
TORCH_MODULE(FluxDiTModel);

REGISTER_MODEL_ARGS(FluxTransformer2DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(dit_patch_size, "patch_size", 1);
  LOAD_ARG_OR(dit_in_channels, "in_channels", 64);
  LOAD_ARG_OR(dit_num_layers, "num_layers", 19);
  LOAD_ARG_OR(dit_num_single_layers, "num_single_layers", 38);
  LOAD_ARG_OR(dit_attention_head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(dit_num_attention_heads, "num_attention_heads", 24);
  LOAD_ARG_OR(dit_joint_attention_dim, "joint_attention_dim", 4096);
  LOAD_ARG_OR(dit_pooled_projection_dim, "pooled_projection_dim", 768);
  LOAD_ARG_OR(dit_guidance_embeds, "guidance_embeds", true);
  LOAD_ARG_OR(
      dit_axes_dims_rope, "axes_dims_rope", (std::vector<int64_t>{16, 56, 56}));
});
}  // namespace xllm
