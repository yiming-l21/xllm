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

#include "add_batch_matmul.h"
namespace xllm_ops {

namespace F = torch::nn::functional;

AddBatchMatmulImpl::AddBatchMatmulImpl(
    int64_t in,
    int64_t out,
    int64_t batch,
    bool with_bias = true,
    const at::Device& device = torch::kCPU,
    const at::ScalarType& dtype = torch::kBFloat16)
    : batch_(batch), with_bias_(with_bias), device_(device), dtype_(dtype) {
  weight_ = register_parameter("weight", torch::empty({batch, out, in}));
  if (with_bias) {
    bias_ = register_parameter("bias", torch::empty({batch, out}));
  } else {
    bias_ = register_parameter("bias", {}, false);
  }
}

torch::Tensor AddBatchMatmulImpl::forward(const torch::Tensor& x) {
  if (with_bias_) {
    return torch::addbmm(bias_, x, weight_, 1, 1);
  } else {
    return torch::bmm(x, weight_);
  }
}

void AddBatchMatmulImpl::load_state_dict(std::vector<torch::Tensor> weights,
                                         std::vector<torch::Tensor> bias) {
  if (weights.size() > 0 && weights[0].defined()) {
    auto batch_weights = torch::stack(weights, 0);
    DCHECK_EQ(weight_.sizes(), batch_weights.sizes())
        << "weight size mismatch: expected " << weight_.sizes() << " but got "
        << batch_weights.sizes();
    weight_.data().copy_(batch_weights);
    weight_.data().to(dtype_).to(device_);
  }
  if (with_bias_ && bias.size() > 0 && bias[0].defined()) {
    auto batch_bias = torch::stack(bias, 0);

    DCHECK_EQ(bias_.sizes(), batch_bias.sizes())
        << "bias size mismatch: expected " << bias_.sizes() << " but got "
        << batch_bias.sizes();
    bias_.data().copy_(batch_bias);
    bias_.data().to(dtype_).to(device_);
  }
}

}  // namespace xllm_ops
