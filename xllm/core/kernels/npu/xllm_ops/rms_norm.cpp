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

#include "rms_norm.h"

#define CHECK_ACL_SUCCESS(expr, msg) \
  do {                               \
    auto _ret = (expr);              \
    if (_ret != ACL_SUCCESS) {       \
      LOG(ERROR) << msg;             \
      LOG(ERROR) << _ret;            \
      throw std::runtime_error(msg); \
    }                                \
  } while (0)

namespace xllm_ops {

RMSNormImpl::RMSNormImpl(int64_t dim,
                         double eps,
                         bool elementwise_affine = true,
                         bool bias = false,
                         const at::Device& device = torch::kCPU,
                         const at::ScalarType& dtype = torch::kBFloat16)
    : eps_(eps),
      elementwise_affine_(elementwise_affine),
      is_bias_(bias),
      device_(device),
      dtype_(dtype) {
  if (elementwise_affine_) {
    weight_ =
        register_parameter("weight", torch::ones({dim}, device_).to(dtype_));
    if (is_bias_) {
      bias_ =
          register_parameter("bias", torch::zeros({dim}, device_).to(dtype_));
    }
  }
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& hidden_states) {
  auto [output, rstd] =
      at_npu::native::custom_ops::npu_rms_norm(hidden_states, weight_, eps_);
  if (is_bias_ && bias_.defined()) {
    output = output + bias_.to(output.device());
  }
  return output;
}

void RMSNormImpl::load_state_dict(const xllm::StateDict& state_dict) {
  if (elementwise_affine_) {
    auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      DCHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch: expected " << weight_.sizes() << " but got "
          << weight.sizes();
      weight_.data().copy_(weight);
      weight_.data().to(dtype_).to(device_);
    }
    if (is_bias_) {
      auto bias = state_dict.get_tensor("bias");
      if (bias.defined()) {
        DCHECK_EQ(bias_.sizes(), bias.sizes())
            << "bias size mismatch: expected " << bias_.sizes() << " but got "
            << bias.sizes();
        bias_.data().copy_(bias);
        bias_.data().to(dtype_).to(device_);
      }
    }
  }
}

}  // namespace xllm_ops
