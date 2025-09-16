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

namespace rope_utils {
std::tuple<torch::Tensor, torch::Tensor> reshape_for_broadcast(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    bool head_first = false);
}  // namespcae rope_utils

namespace xllm_ops {
torch::Tensor rotary_position_embedding(const torch::Tensor x, const torch::Tensor cos, const torch::Tensor sin,
                        string rotated_mode, bool head_first, bool fused);
}  // namespace xllm_ops
