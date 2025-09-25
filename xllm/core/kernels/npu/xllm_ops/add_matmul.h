#pragma once
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "core/framework/state_dict/state_dict.h"
#include "util/tensor_helper.h"

namespace xllm_ops {

class DiTLinearImpl : public torch::nn::Module {
 public:
  DiTLinearImpl(int64_t in,
                int64_t out,
                bool with_bias,
                torch::TensorOptions options = torch::TensorOptions()
                                                   .dtype(torch::kBFloat16)
                                                   .device(torch::kCPU));

  torch::Tensor forward(const torch::Tensor& x);

  void load_state_dict(const xllm::StateDict& state_dict);

  torch::Tensor weight_;
  torch::Tensor bias_;
  bool with_bias_;
  torch::TensorOptions options_;
};

TORCH_MODULE(DiTLinear);

}  // namespace xllm_ops
