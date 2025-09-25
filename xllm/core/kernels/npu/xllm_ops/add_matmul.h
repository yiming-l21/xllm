#pragma once
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "core/framework/state_dict/state_dict.h"
#include "util/tensor_helper.h"

namespace xllm_ops {

class AddMatmulImpl : public torch::nn::Module {
 public:
  AddMatmulImpl(int64_t in,
                int64_t out,
                bool with_bias,
                const at::Device& device,
                const at::ScalarType& dtype);

  torch::Tensor forward(const torch::Tensor& x);

  void load_state_dict(const xllm::StateDict& state_dict);

  torch::Tensor weight_;
  torch::Tensor bias_;
  bool with_bias_;
  at::Device device_;
  at::ScalarType dtype_;
};

TORCH_MODULE(AddMatmul);

}  // namespace xllm_ops
