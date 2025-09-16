#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm.h"
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
    int32_t device_id = hidden_states.device().index();
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

    aclTensor* acl_hidden_states = nullptr;
    aclTensor* acl_gamma = nullptr;
    aclTensor* acl_output = nullptr;
    aclTensor* acl_rstd = nullptr;

    auto input_shape = hidden_states.sizes();
    std::vector<int64_t> rstd_shape;
    for (int64_t i = 0; i < hidden_states.dim() - 1; ++i) {
        rstd_shape.push_back(input_shape[i]);
    }
    rstd_shape.push_back(1);  // keepdim=true

    auto rstd = torch::empty(rstd_shape, torch::TensorOptions().dtype(torch::kFloat32).device(hidden_states.device()));
    auto output = torch::empty_like(hidden_states);
    torch::Tensor gamma;
    if (elementwise_affine_) {
        gamma = weight_;
    } else {
        int64_t hidden_size = input_shape[input_shape.size() - 1];
        gamma = torch::ones({hidden_size},
                           hidden_states.options().dtype(hidden_states.dtype()).device(hidden_states.device()));
    }
    xllm_ops_utils::create_acltensor(&acl_hidden_states, hidden_states);
    xllm_ops_utils::create_acltensor(&acl_gamma, gamma);
    xllm_ops_utils::create_acltensor(&acl_output, output);
    xllm_ops_utils::create_acltensor(&acl_rstd, rstd);
    aclOpExecutor* executor;
    uint64_t workspace_size = 0;

    CHECK_ACL_SUCCESS(aclnnRmsNormGetWorkspaceSize(acl_hidden_states, acl_gamma, eps_, acl_output, acl_rstd, &workspace_size, &executor),
                      "RMSNORM: failed to get workspace size");
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        CHECK_ACL_SUCCESS(aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
                         "RMSNORM: failed to malloc memory");
    }
    CHECK_ACL_SUCCESS(aclnnRmsNorm(workspace_addr, workspace_size, executor, stream),
                      "RMSNORM: execute failed");
    CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                      "RMSNORM: sychronize failed");
    aclDestroyTensor(acl_hidden_states);
    aclDestroyTensor(acl_gamma);
    aclDestroyTensor(acl_output);
    aclDestroyTensor(acl_rstd);
    aclrtFree(workspace_addr);
    if (is_bias_) {
        output = output + bias_.to(output.device());
    }
    return output;
  }
  
  void RMSNormImpl::load_state_dict(const xllm::StateDict& state_dict) {
    if (elementwise_affine_) {
      auto weight = state_dict.get_tensor("weight");
      if (weight.defined()) {
        DCHECK_EQ(weight_.sizes(), weight.sizes())
            << "weight size mismatch: expected " << weight_.sizes()
            << " but got " << weight.sizes();
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

  
} // namespace xllm_ops
