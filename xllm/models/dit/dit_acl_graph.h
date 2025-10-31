#pragma once

#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>

#include "core/runtime/dit_forward_params.h"

namespace xllm {

class CLIPTextModel;
class T5EncoderModel;
class VAE;
class FlowMatchEulerDiscreteScheduler;
class FluxTransformerBlock;
class FluxSingleTransformerBlock;

class CLIPAclGraph {
 public:
  CLIPAclGraph() = default;

  torch::Tensor capture(CLIPTextModel& model,
                        const torch::TensorOptions& options,
                        torch::Tensor input_ids);

  torch::Tensor replay(torch::Tensor input_ids);

 private:
  c10_npu::NPUGraph graph_;

  // input
  torch::Tensor input_ids_;

  // output
  torch::Tensor output_;
};

class T5AclGraph {
 public:
  T5AclGraph() = default;

  torch::Tensor capture(T5EncoderModel& model,
                        const torch::TensorOptions& options,
                        torch::Tensor input_ids);

  torch::Tensor replay(torch::Tensor input_ids);

 private:
  c10_npu::NPUGraph graph_;

  // input
  torch::Tensor input_ids_;

  // output
  torch::Tensor output_;
};

class VAEAclGraph {
 public:
  VAEAclGraph() = default;

  torch::Tensor capture(VAE& model,
                        const torch::TensorOptions& options,
                        torch::Tensor latents);

  torch::Tensor replay(torch::Tensor latents);

 private:
  c10_npu::NPUGraph graph_;

  // input
  torch::Tensor latents_;

  // output
  torch::Tensor output_;
};

class SchedulerAclGraph {
 public:
  SchedulerAclGraph() = default;

  torch::Tensor capture(FlowMatchEulerDiscreteScheduler& model,
                        const torch::TensorOptions& options,
                        torch::Tensor noise_pred,
                        torch::Tensor t,
                        torch::Tensor prepared_latents);

  torch::Tensor replay(torch::Tensor noise_pred,
                       torch::Tensor t,
                       torch::Tensor prepared_latents);

 private:
  c10_npu::NPUGraph graph_;

  // input
  torch::Tensor noise_pred_;
  torch::Tensor t_;
  torch::Tensor prepared_latents_;

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
