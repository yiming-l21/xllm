#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/libs/init_npu.h>

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "add_matmul.h"

namespace xllm_ops {

namespace F = torch::nn::functional;

DiTLinearImpl::DiTLinearImpl(int64_t in,
                             int64_t out,
                             bool with_bias,
                             torch::TensorOptions options)
    : with_bias_(with_bias), options_(options) {
  weight_ = register_parameter("weight", torch::empty({out, in}, options_));
  if (with_bias) {
    bias_ = register_parameter("bias", torch::empty(out, options_));
  } else {
    bias_ = register_parameter("bias", {}, false);
  }
}

torch::Tensor DiTLinearImpl::forward(const torch::Tensor& x) {
  if (with_bias_) {
    auto sizes = x.sizes();
    if (sizes.size() == 3) {
      torch::Tensor x_;
      x_ = x.reshape({sizes[0] * sizes[1], sizes[2]});
      return torch::addmm(bias_, x_, weight_, 1, 1)
          .reshape({sizes[0], sizes[1], weight_.size(1)});
    } else {
      return torch::addmm(bias_, x, weight_, 1, 1);
    }
  } else {
    return F::linear(x, weight_, bias_);
  }
}

void DiTLinearImpl::load_state_dict(const xllm::StateDict& state_dict) {
  auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    DCHECK_EQ(weight_.sizes(), weight.sizes())
        << "weight size mismatch: expected " << weight_.sizes() << " but got "
        << weight.sizes();
    weight_.data().copy_(weight);
    weight_.data().to(options_);
    if (with_bias_) {
      weight_ = weight_.transpose(0, 1).contiguous();
    }
  }
  if (with_bias_) {
    auto bias = state_dict.get_tensor("bias");
    if (bias.defined()) {
      DCHECK_EQ(bias_.sizes(), bias.sizes())
          << "bias size mismatch: expected " << bias_.sizes() << " but got "
          << bias.sizes();
      bias_.data().copy_(bias);
      bias_.data().to(options_);
    }
  }
}

}  // namespace xllm_ops
