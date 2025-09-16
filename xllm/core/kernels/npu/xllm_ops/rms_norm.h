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
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "acltensor_utils.h"
#include "util/tensor_helper.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm_ops {

class RMSNormImpl : public torch::nn::Module {
  public:
    // Constructor: dim (normalization dimension), eps (stabilization term)
    // elementwise_affine (enable affine transform), bias (enable bias term)
    RMSNormImpl(int64_t dim,
	        double eps,
		bool elementwise_affine,
		bool bias,
		const at::Device& device,
		const at::ScalarType& dtype);
    
    torch::Tensor forward(const torch::Tensor& hidden_states);
    
    void load_state_dict(const xllm::StateDict& state_dict);  

  private:
    float eps_;                // Small epsilon to avoid division by zero
    bool elementwise_affine_;  // Whether to apply learnable affine parameters
    torch::Tensor weight_;     // Learnable scale parameter
    torch::Tensor bias_;       // Learnable bias parameter (optional)
    bool is_bias_;
    at::Device device_;
    at::ScalarType dtype_;  // Data type for the parameters
};

TORCH_MODULE(RMSNorm);

}  // namespace xllm_ops

