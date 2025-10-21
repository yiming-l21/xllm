#include "dit_acl_graph.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include "dit.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {

DiTAclGraph::DiTAclGraph() : model_(nullptr) {}

DiTAclGraph::~DiTAclGraph() {
  if (model_ != nullptr) aclmdlRIDestroy(model_);
}

void DiTAclGraph::capture(const DiTForwardInput& input,
                          FluxDiTModel& model,
                          const torch::TensorOptions& options) {
  CHECK(model_ == nullptr) << "graph already captured";

  hidden_states_ = torch::zeros({1, 1024, 64}, options);
  encoder_hidden_states_ = torch::zeros({1, 512, 4096}, options);

  pooled_projections_ = torch::zeros({1, 768}, options);
  timestep_ = torch::zeros({1}, options);

  image_rotary_emb_ = torch::zeros({2, 1536, 128}, options);
  guidance_ = torch::zeros({1}, options);

  torch::npu::synchronize();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(options.device().index()).stream();

  // aclError st = aclmdlRICaptureBegin(stream,
  // ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
  aclError st = aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
  CHECK_EQ(st, ACL_SUCCESS)
      << "aclmdlRICaptureBegin failed, error code: " << st;

  output_ = model->forward(hidden_states_,
                           encoder_hidden_states_,
                           pooled_projections_,
                           timestep_,
                           image_rotary_emb_,
                           guidance_,
                           0);

  st = aclmdlRICaptureEnd(stream, &model_);
  CHECK_EQ(st, ACL_SUCCESS) << "aclmdlRICaptureEnd failed, error code: " << st;
  torch::npu::synchronize();
}

torch::Tensor DiTAclGraph::replay(torch::Tensor hidden_states,
                                  torch::Tensor encoder_hidden_states,
                                  torch::Tensor pooled_projections,
                                  torch::Tensor timestep,
                                  torch::Tensor image_rotary_emb,
                                  torch::Tensor guidance,
                                  int64_t step_idx) {
  CHECK(model_ != nullptr) << "graph not captured";

  hidden_states_.copy_(hidden_states, true);
  encoder_hidden_states_.copy_(encoder_hidden_states, true);

  pooled_projections_.copy_(pooled_projections, true);
  timestep_.copy_(timestep, true);

  image_rotary_emb_.copy_(image_rotary_emb, true);
  guidance_.copy_(guidance, true);

  // replay the graph
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  aclError st = aclmdlRIExecuteAsync(model_, stream);

  CHECK_EQ(st, ACL_SUCCESS)
      << "aclmdlRIExecuteAsync failed, error code: " << st;

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
