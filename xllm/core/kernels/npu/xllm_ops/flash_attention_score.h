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

namespace xllm_ops {
torch::Tensor flash_attention_score(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value, const c10::optional<torch::Tensor>& attn_mask,
         const c10::optional<torch::Tensor>& drop_mask, double keep_prob, bool is_causal, double scale, const string layout);
}  // namespace xllm_ops
