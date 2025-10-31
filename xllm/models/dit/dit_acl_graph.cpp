#include "dit_acl_graph.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include "autoencoder_kl.h"
#include "clip_text_model.h"
#include "dit.h"
#include "flowmatch_euler_discrete_scheduler.h"
#include "t5_encoder.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {

torch::Tensor CLIPAclGraph::capture(CLIPTextModel& model,
                                    const torch::TensorOptions& options,
                                    torch::Tensor input_ids) {
  // input tensor
  input_ids_ = torch::zeros({1, 77}, options.dtype(torch::kInt64));

  // output tensor
  output_ = torch::zeros({1, 768}, options);

  torch::npu::synchronize();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(options.device().index()).stream();

  input_ids_.copy_(input_ids, true);

  aclrtSynchronizeStream(stream);

  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(options.device().index()) ==
      c10_npu::getDefaultNPUStream(options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }

  graph_.capture_begin();
  output_ = model->forward(input_ids_);
  graph_.capture_end();

  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(options.device().index()));
  }

  torch::npu::synchronize();
  graph_.replay();

  return output_;
}

torch::Tensor CLIPAclGraph::replay(torch::Tensor input_ids) {
  input_ids_.copy_(input_ids, true);

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  return output_;
}

torch::Tensor T5AclGraph::capture(T5EncoderModel& model,
                                  const torch::TensorOptions& options,
                                  torch::Tensor input_ids) {
  // input tensor
  input_ids_ = torch::zeros({1, 512}, options.dtype(torch::kInt64));

  // output tensor
  output_ = torch::zeros({1, 512, 4096}, options);

  torch::npu::synchronize();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(options.device().index()).stream();

  input_ids_.copy_(input_ids, true);

  aclrtSynchronizeStream(stream);

  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(options.device().index()) ==
      c10_npu::getDefaultNPUStream(options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }

  graph_.capture_begin();
  output_ = model->forward(input_ids_);
  graph_.capture_end();

  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(options.device().index()));
  }

  torch::npu::synchronize();
  graph_.replay();

  return output_;
}

torch::Tensor T5AclGraph::replay(torch::Tensor input_ids) {
  input_ids_.copy_(input_ids, true);

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  return output_;
}

torch::Tensor VAEAclGraph::capture(VAE& model,
                                   const torch::TensorOptions& options,
                                   torch::Tensor latents) {
  // input tensor
  latents_ = torch::zeros({1, 16, 64, 64}, options);

  // output tensor
  output_ = torch::zeros({1, 3, 512, 512}, options);

  torch::npu::synchronize();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(options.device().index()).stream();

  latents_.copy_(latents, true);

  aclrtSynchronizeStream(stream);

  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(options.device().index()) ==
      c10_npu::getDefaultNPUStream(options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }

  graph_.capture_begin();
  output_ = model->decode(latents_).sample;
  graph_.capture_end();

  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(options.device().index()));
  }

  torch::npu::synchronize();
  graph_.replay();

  return output_;
}

torch::Tensor VAEAclGraph::replay(torch::Tensor latents) {
  latents_.copy_(latents, true);

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  return output_;
}

torch::Tensor SchedulerAclGraph::capture(FlowMatchEulerDiscreteScheduler& model,
                                         const torch::TensorOptions& options,
                                         torch::Tensor noise_pred,
                                         torch::Tensor t,
                                         torch::Tensor prepared_latents) {
  // input tensor
  noise_pred_ = torch::zeros({1, 1024, 64}, options);
  t_ = torch::zeros({1}, options.dtype(torch::kFloat));
  prepared_latents_ = torch::zeros({1, 1024, 64}, options);

  // output tensor
  output_ = torch::zeros({1, 1024, 64}, options);

  torch::npu::synchronize();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(options.device().index()).stream();

  noise_pred_.copy_(noise_pred, true);
  t_.copy_(t, true);
  prepared_latents_.copy_(prepared_latents, true);

  aclrtSynchronizeStream(stream);

  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(options.device().index()) ==
      c10_npu::getDefaultNPUStream(options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }

  graph_.capture_begin();
  output_ = model->step(noise_pred_, t_, prepared_latents_).prev_sample;
  graph_.capture_end();

  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(options.device().index()));
  }

  torch::npu::synchronize();
  graph_.replay();

  return output_;
}

torch::Tensor SchedulerAclGraph::replay(torch::Tensor noise_pred,
                                        torch::Tensor t,
                                        torch::Tensor prepared_latents) {
  noise_pred_.copy_(noise_pred, true);
  t_.copy_(t, true);
  prepared_latents_.copy_(prepared_latents, true);

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  return output_;
}

TransBlockAclGraph::TransBlockAclGraph() {}

TransBlockAclGraph::~TransBlockAclGraph() {}

std::tuple<torch::Tensor, torch::Tensor> TransBlockAclGraph::capture(
    FluxTransformerBlock& model,
    const torch::TensorOptions& options,
    torch::Tensor hidden_states,
    torch::Tensor encoder_hidden_states,
    torch::Tensor temb,
    torch::Tensor image_rotary_emb) {
  // input tensor
  hidden_states_ = torch::zeros({1, 1024, 3072}, options);
  encoder_hidden_states_ = torch::zeros({1, 512, 3072}, options);

  temb_ = torch::zeros({1, 3072}, options);
  image_rotary_emb_ = torch::zeros({2, 1536, 128}, options);

  // output tensor
  std::get<0>(output_) = torch::zeros({1, 1024, 3072}, options);
  std::get<1>(output_) = torch::zeros({1, 512, 3072}, options);

  torch::npu::synchronize();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(options.device().index()).stream();

  hidden_states_.copy_(hidden_states, true);
  encoder_hidden_states_.copy_(encoder_hidden_states, true);

  temb_.copy_(temb, true);
  image_rotary_emb_.copy_(image_rotary_emb, true);

  aclrtSynchronizeStream(stream);

  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(options.device().index()) ==
      c10_npu::getDefaultNPUStream(options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }

  graph_.capture_begin();
  output_ = model->forward(
      hidden_states_, encoder_hidden_states_, temb_, image_rotary_emb_);
  graph_.capture_end();

  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(options.device().index()));
  }

  torch::npu::synchronize();
  graph_.replay();

  return output_;
}

std::tuple<torch::Tensor, torch::Tensor> TransBlockAclGraph::replay(
    torch::Tensor hidden_states,
    torch::Tensor encoder_hidden_states,
    torch::Tensor temb,
    torch::Tensor image_rotary_emb) {
  hidden_states_.copy_(hidden_states, true);
  encoder_hidden_states_.copy_(encoder_hidden_states, true);

  temb_.copy_(temb, true);
  image_rotary_emb_.copy_(image_rotary_emb, true);

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  return output_;
}

SigTransBlockAclGraph::SigTransBlockAclGraph() {}

SigTransBlockAclGraph::~SigTransBlockAclGraph() {}

void SigTransBlockAclGraph::capture(FluxSingleTransformerBlock& model,
                                    const torch::TensorOptions& options) {
  hidden_states_ = torch::zeros({1, 1024 + 512, 3072}, options);

  temb_ = torch::zeros({1, 3072}, options);
  image_rotary_emb_ = torch::zeros({2, 1536, 128}, options);

  torch::npu::synchronize();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(options.device().index()).stream();

  aclrtSynchronizeStream(stream);

  bool need_restore_stream = false;
  if (c10_npu::getCurrentNPUStream(options.device().index()) ==
      c10_npu::getDefaultNPUStream(options.device().index())) {
    auto secondary_stream =
        c10_npu::getStreamFromPool(true, options.device().index());
    c10_npu::setCurrentNPUStream(secondary_stream);
    need_restore_stream = true;
  }

  graph_.capture_begin();
  output_ = model->forward(hidden_states_, temb_, image_rotary_emb_);
  graph_.capture_end();

  if (need_restore_stream) {
    c10_npu::setCurrentNPUStream(
        c10_npu::getDefaultNPUStream(options.device().index()));
  }

  aclrtSynchronizeStream(stream);
  graph_.replay();
}

torch::Tensor SigTransBlockAclGraph::replay(torch::Tensor hidden_states,
                                            torch::Tensor temb,
                                            torch::Tensor image_rotary_emb) {
  hidden_states_.copy_(hidden_states, true);

  temb_.copy_(temb, true);
  image_rotary_emb_.copy_(image_rotary_emb, true);

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  // replay the graph
  graph_.replay();

  return output_;
}

}  // namespace xllm
