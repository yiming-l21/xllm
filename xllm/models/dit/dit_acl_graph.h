#pragma once

#include <acl/acl.h>
#include <torch/torch.h>

#include "core/runtime/dit_forward_params.h"
#include "dit.h"

namespace xllm {

class DiTAclGraph {
 public:
  DiTAclGraph() : model_(nullptr) {}

  void capture(const DiTForwardInput& input,
               hf::FluxDiTModel& model,
               torch::ScalarType dtype,
               torch::Device device);

  torch::Tensor replay(torch::Tensor hidden_states,
                       torch::Tensor encoder_hidden_states,
                       torch::Tensor pooled_projections,
                       torch::Tensor timestep,
                       torch::Tensor img_ids,
                       torch::Tensor txt_ids,
                       torch::Tensor guidance,
                       int64_t step_idx = 0);

 private:
  aclmdlRI model_;

  // input
  torch::Tensor hidden_states_;
  torch::Tensor encoder_hidden_states_;

  torch::Tensor pooled_projections_;
  torch::Tensor timestep_;

  torch::Tensor img_ids_;
  torch::Tensor txt_ids_;

  torch::Tensor guidance_;

  // output
  torch::Tensor output_;
};

}  // namespace xllm
