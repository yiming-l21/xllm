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
#include "aclnnop/aclnn_flash_attention_score.h"
#include "flash_attention_score.h"

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
  torch::Tensor flash_attention_score(const torch::Tensor& query, const torch::Tensor& key, const torch::Tensor& value, const c10::optional<torch::Tensor>& attn_mask=torch::nullopt, 
         const c10::optional<torch::Tensor>& drop_mask=torch::nullopt, double keep_prob=1.0, bool is_causal=false, double scale=0.0, const string layout="BNSD") {
    
    aclTensor* acl_query = nullptr;
    aclTensor* acl_key = nullptr;
    aclTensor* acl_value = nullptr;
    aclTensor* acl_attn_mask = nullptr;
    aclTensor* acl_attention_out = nullptr;
    aclTensor* acl_drop_mask = nullptr;
    auto device = query.device();
    auto dtype = query.dtype();
    
    torch::Tensor attention_out = torch::empty_like(query);

    int32_t device_id = query.device().index();
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

    xllm_ops_utils::create_acltensor(&acl_query, query);
    xllm_ops_utils::create_acltensor(&acl_key, key);
    xllm_ops_utils::create_acltensor(&acl_value, value);
    xllm_ops_utils::create_acltensor(&acl_attention_out, attention_out);
    if(attn_mask.has_value()){
        auto attn_mask_tensor = attn_mask.value();
        xllm_ops_utils::create_acltensor(&acl_attn_mask, attn_mask_tensor);
    }
    if(drop_mask.has_value()){
        auto drop_mask_tensor = drop_mask.value();
        xllm_ops_utils::create_acltensor(&acl_drop_mask, drop_mask_tensor);
    }

    aclTensor* acl_pse = nullptr;
    aclTensor* acl_padding = nullptr;
    aclTensor* acl_softmaxMax = nullptr;
    aclTensor* acl_softmaxSum = nullptr;
    aclTensor* acl_softmaxOut = nullptr;
    
    std::vector<int64_t> prefixOp = {0};
    aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
    int64_t pre_tokens = 65536;
    int64_t next_tokens = 65536;
    int64_t head_num = (layout == std::string("BNSD")) ? query.size(1) : query.size(2);
    int64_t seq_len = (layout == std::string("BNSD")) ? query.size(2) : query.size(1);
    int64_t inner_precise = 0;
    int64_t sparse_mod = 0;
    
    auto softmaxMax = torch::empty({query.size(0), head_num, seq_len, 8},
            torch::TensorOptions()
                .dtype(torch::kFloat)
                .device(query.device())
        );
    auto softmaxSum = torch::empty({query.size(0), head_num, seq_len, 8},
            torch::TensorOptions()
                .dtype(torch::kFloat)
                .device(query.device())
        );
    xllm_ops_utils::create_acltensor(&acl_softmaxMax, softmaxMax);
    xllm_ops_utils::create_acltensor(&acl_softmaxSum, softmaxSum);
    
    char* layout_ = new char[layout.length() + 1];
    std::strcpy(layout_, layout.c_str());

    uint64_t workspace_size = 0;
    aclOpExecutor* executor;
    CHECK_ACL_SUCCESS(aclnnFlashAttentionScoreGetWorkspaceSize(
              acl_query, acl_key, acl_value, acl_pse, acl_drop_mask, acl_padding, acl_attn_mask, prefix, scale,
              keep_prob, pre_tokens, next_tokens, head_num, layout_, inner_precise,
              sparse_mod, acl_softmaxMax, acl_softmaxSum, acl_softmaxOut, acl_attention_out, &workspace_size, &executor),
              "flash_attention_score: failed to get workspace size");

     
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
      CHECK_ACL_SUCCESS(aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
              "flash_attention_score: failed to malloc memory");
    }
    
    CHECK_ACL_SUCCESS(aclnnFlashAttentionScore(workspace_addr, workspace_size, executor, stream),
              "flash_attention_score execute failed");
    
    CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
              "flash_attention_score sychronize failed");
    
    aclDestroyTensor(acl_query);
    aclDestroyTensor(acl_key);
    aclDestroyTensor(acl_value);
    aclDestroyTensor(acl_attn_mask);
    aclDestroyTensor(acl_drop_mask);
    aclDestroyTensor(acl_attention_out);
    aclDestroyTensor(acl_pse);
    aclDestroyTensor(acl_padding);
    aclDestroyTensor(acl_softmaxMax);
    aclDestroyTensor(acl_softmaxSum);
    aclDestroyTensor(acl_softmaxOut);
    aclrtFree(workspace_addr);    
    
    return attention_out;
  }
}  // namespace xllm_ops
