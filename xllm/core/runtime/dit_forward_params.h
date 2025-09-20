/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <nlohmann/json.hpp>
#include <optional>

#include "framework/request/dit_request_state.h"

namespace xllm {

// dit related forward input params
struct DiTForwardInput {
  DiTForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype) const {
    DiTForwardInput input;
    input.input_params = input_params.to(device, dtype);
    input.generation_params = generation_params;
    return input;
  }

  DiTInputParams input_params;
  DiTGenerationParams generation_params;
};

// dit related forward output params
struct DiTForwardOutput {
  // generated tensor
  torch::Tensor tensor;

  DiTForwardOutput slice(int offset, int count) const {
    DiTForwardOutput output;
    output.tensor = tensor.slice(0, offset, offset + count).clone();

    return output;
  }
};

}  // namespace xllm
