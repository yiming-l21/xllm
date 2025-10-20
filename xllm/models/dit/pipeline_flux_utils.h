/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <acl/acl.h>
#include <torch/torch.h>

#include <cmath>
#include <optional>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "flowmatch_euler_discrete_scheduler.h"
#include "models/model_registry.h"

namespace xllm {
float calculate_shift(int64_t image_seq_len,
                      int64_t base_seq_len = 256,
                      int64_t max_seq_len = 4096,
                      float base_shift = 0.5f,
                      float max_shift = 1.15f) {
  float m =
      (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
  float b = base_shift - m * static_cast<float>(base_seq_len);
  float mu = static_cast<float>(image_seq_len) * m + b;
  return mu;
}

std::pair<torch::Tensor, int64_t> retrieve_timesteps(
    FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(
        static_cast<int>(steps), device, std::nullopt, mu, std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (scheduler_timesteps.device() != device) {
    scheduler_timesteps = scheduler_timesteps.to(device);
  }
  return {scheduler_timesteps, steps};
}

torch::Tensor randn_tensor(const std::vector<int64_t>& shape,
                           int64_t seed,
                           torch::TensorOptions& options) {
  if (shape.empty()) {
    LOG(FATAL) << "Shape must not be empty.";
  }
  torch::manual_seed(seed);
  torch::Tensor latents;
  latents = torch::randn(shape, options);
  return latents;
}

torch::Tensor get_1d_rotary_pos_embed(
    int64_t dim,
    const torch::Tensor& pos,
    float theta = 10000.0,
    bool use_real = false,
    float linear_factor = 1.0,
    float ntk_factor = 1.0,
    bool repeat_interleave_real = true,
    torch::Dtype freqs_dtype = torch::kFloat32) {
  TORCH_CHECK(dim % 2 == 0, "Dimension must be even");

  torch::Tensor pos_tensor = pos;
  if (pos.dim() == 0) {
    pos_tensor = torch::arange(pos.item<int64_t>(), pos.options());
  }

  theta = theta * ntk_factor;

  auto freqs =
      1.0 /
      (torch::pow(
           theta,
           torch::arange(
               0, dim, 2, torch::dtype(freqs_dtype).device(pos.device())) /
               dim) *
       linear_factor);  // [D/2]

  auto tensors = {pos_tensor, freqs};

  auto freqs_outer = torch::einsum("s,d->sd", tensors);  // [S, D/2]
#if defined(USE_NPU)
  freqs_outer = freqs_outer.to(torch::kFloat32);
#endif
  if (use_real && repeat_interleave_real) {
    auto cos_vals = torch::cos(freqs_outer);  // [S, D/2]
    auto sin_vals = torch::sin(freqs_outer);  // [S, D/2]

    auto freqs_cos = cos_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]

    auto freqs_sin = sin_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]
    return torch::cat({freqs_cos.unsqueeze(0), freqs_sin.unsqueeze(0)},
                      0);  // [2, S, D]
  }
}

class FluxPosEmbedImpl : public torch::nn::Module {
 public:
  FluxPosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }

  std::pair<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& txt_ids,
      const torch::Tensor& img_ids,
      int64_t height = -1,
      int64_t width = -1) {
    auto seq_len = txt_ids.size(0);

    // recompute the cache if height or width changes
    if (height != cached_image_height_ || width != cached_image_width_ ||
        seq_len != max_seq_len_) {
      torch::Tensor ids = torch::cat({txt_ids, img_ids}, 0);
      cached_image_height_ = height;
      cached_image_width_ = width;
      max_seq_len_ = seq_len;
      auto [cos, sin] = forward(ids);
      freqs_cos_cache_ = std::move(cos);
      freqs_sin_cache_ = std::move(sin);
    }
    return {freqs_cos_cache_, freqs_sin_cache_};
  }

  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
    int64_t n_axes = ids.size(-1);
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    torch::Dtype freqs_dtype = torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      auto pos_slice = pos.select(-1, i);
      auto result = get_1d_rotary_pos_embed(axes_dim_[i],
                                            pos_slice,
                                            theta_,
                                            true,  // repeat_interleave_real
                                            1,
                                            1,
                                            true,  // use_real
                                            freqs_dtype);
      auto cos = result[0];
      auto sin = result[1];
      cos_out.push_back(cos);
      sin_out.push_back(sin);
    }

    auto freqs_cos = torch::cat(cos_out, -1);
    auto freqs_sin = torch::cat(sin_out, -1);
    return {freqs_cos, freqs_sin};
  }

 private:
  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  torch::Tensor freqs_cos_cache_;
  torch::Tensor freqs_sin_cache_;
  int64_t max_seq_len_ = -1;
  int64_t cached_image_height_ = -1;
  int64_t cached_image_width_ = -1;
};
TORCH_MODULE(FluxPosEmbed);

}  // namespace xllm