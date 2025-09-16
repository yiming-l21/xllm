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

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "rotary_embedding.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"

namespcae rope_utils{
   std::tuple<torch::Tensor, torch::Tensor> reshape_for_broadcast(
        const torch::Tensor& x, 
        const torch::Tensor& cos, 
        const torch::Tensor& sin, 
        bool head_first) {
        
        auto ndim = x.dim();
        std::vector<int64_t> shape;
        shape.reserve(ndim);
        
        if (head_first) {
            for (int64_t i = 0; i < ndim; ++i) {
                if (i == ndim - 2 || i == ndim - 1) {
                    shape.push_back(x.size(i));
                } else {
                    shape.push_back(1);
                }
            }
        } else {
            for (int64_t i = 0; i < ndim; ++i) {
                if (i == 1 || i == ndim - 1) {
                    shape.push_back(x.size(i));
                } else {
                    shape.push_back(1);
                }
            }
        }
        
        return std::make_tuple(cos.view(shape), sin.view(shape));
    }

}
namespace xllm_ops {
    
    torch::Tensor rotary_position_embedding(const torch::Tensor x, const torch::Tensor cos, const torch::Tensor sin, 
                        string rotated_mode="rotated_interleaved", bool head_first=False, bool fused=True){
    if (cos.dim() == 2 && sin.dim() == 2) {
            auto [reshaped_cos, reshaped_sin] = 
                TensorUtils::reshape_for_broadcast(x, cos, sin, head_first);
       }
    
    int64_t mode = 0;
    if (rotated_mode == std::string("rotated_half")){
        mode = 0; 
    } else {
        mode = 1;
    }
 
    int32_t device_id = x.device().index();
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
    
    auto device = x.device();
    auto dtype = x.dtype();

    torch::Tensor out = torch::empty_like(x);
    
    aclTensor* acl_x = nullptr;
    aclTensor* acl_cos = nullptr;
    aclTensor* acl_sin = nullptr;
    aclTensor* acl_out = nullptr;
    
    xllm_ops_utils::create_acltensor(&acl, acl);
    xllm_ops_utils::create_acltensor(&acl_cos, cos);
    xllm_ops_utils::create_acltensor(&acl_sin, sin);
    xllm_ops_utils::create_acltensor(&acl_out, out);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor;
    CHECK_ACL_SUCCESS(aclnnRotaryPositionEmbeddingGetWorkspaceSize(acl_x, acl_cos, acl_sin, mode, acl_out, &workspace_size, &executor),
                     "rotary_embedding: failed to get workspace size");
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        CHECK_ACL_SUCCESS(aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
                          "rotary_embedding: failed to malloc workspace size")
    }
    CHECK_ACL_SUCCESS(aclnnRotaryPositionEmbedding(workspaceAddr, workspaceSize, executor, stream),
                      "rotary_embedding: failed to execute rotary_embedding");
    CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                      "rotary_embedding: failed to synchronize stream for  rotary_embedding");
    
    aclDestroyTensor(acl_x);
    aclDestroyTensor(acl_cos);
    aclDestroyTensor(acl_sin);
    aclDestroyTensor(acl_out);

    return out;
    }
}

