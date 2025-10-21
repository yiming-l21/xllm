#pragma once

#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>

#include "core/runtime/dit_forward_params.h"

namespace xllm {

class FluxDiTModel;
class FluxTransformerBlock;
class FluxSingleTransformerBlock;

class DiTAclGraph {
 public:
  DiTAclGraph();
  ~DiTAclGraph();

  void capture(const DiTForwardInput& input,
               FluxDiTModel& model,
               const torch::TensorOptions& options);

  torch::Tensor replay(torch::Tensor hidden_states,
                       torch::Tensor encoder_hidden_states,
                       torch::Tensor pooled_projections,
                       torch::Tensor timestep,
                       torch::Tensor image_rotary_emb,
                       torch::Tensor guidance,
                       int64_t step_idx = 0);

 private:
  aclmdlRI model_;

  // input
  torch::Tensor hidden_states_;
  torch::Tensor encoder_hidden_states_;

  torch::Tensor pooled_projections_;
  torch::Tensor timestep_;

  torch::Tensor image_rotary_emb_;
  torch::Tensor guidance_;

  // output
  torch::Tensor output_;
};

class TransBlockAclGraph {
 public:
  TransBlockAclGraph();
  ~TransBlockAclGraph();

  std::tuple<torch::Tensor, torch::Tensor> capture(
      FluxTransformerBlock& model,
      const torch::TensorOptions& options,
      torch::Tensor hidden_states,
      torch::Tensor encoder_hidden_states,
      torch::Tensor temb,
      torch::Tensor image_rotary_emb);

  std::tuple<torch::Tensor, torch::Tensor> replay(
      torch::Tensor hidden_states,
      torch::Tensor encoder_hidden_states,
      torch::Tensor temb,
      torch::Tensor image_rotary_emb);

 private:
  c10_npu::NPUGraph graph_;

  // input
  torch::Tensor hidden_states_;
  torch::Tensor encoder_hidden_states_;

  torch::Tensor temb_;
  torch::Tensor image_rotary_emb_;

  // output
  std::tuple<torch::Tensor, torch::Tensor> output_;
};

class SigTransBlockAclGraph {
 public:
  SigTransBlockAclGraph();
  ~SigTransBlockAclGraph();

  void capture(FluxSingleTransformerBlock& model,
               const torch::TensorOptions& options);

  torch::Tensor replay(torch::Tensor hidden_states,
                       torch::Tensor temb,
                       torch::Tensor image_rotary_emb);

 private:
  c10_npu::NPUGraph graph_;

  // input
  torch::Tensor hidden_states_;

  torch::Tensor temb_;
  torch::Tensor image_rotary_emb_;

  // output
  torch::Tensor output_;
};

}  // namespace xllm
