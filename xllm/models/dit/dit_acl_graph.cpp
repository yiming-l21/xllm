#include "dit_acl_graph.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {

void DiTAclGraph::capture(const DiTForwardInput& input,
                          hf::FluxDiTModel& model,
                          torch::ScalarType dtype,
                          torch::Device device) {
  CHECK(model_ == nullptr) << "graph already captured";

  auto options = torch::dtype(dtype).device(device);

  hidden_states_ = torch::zeros({1, 1024, 64}, options);
  encoder_hidden_states_ = torch::zeros({1, 512, 4096}, options);

  pooled_projections_ = torch::zeros({1, 768}, options);
  timestep_ = torch::zeros({1}, options);

  img_ids_ = torch::zeros({1024, 3}, options);
  txt_ids_ = torch::zeros({512, 3}, options);

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
                           img_ids_,
                           txt_ids_,
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
                                  torch::Tensor img_ids,
                                  torch::Tensor txt_ids,
                                  torch::Tensor guidance,
                                  int64_t step_idx) {
  CHECK(model_ != nullptr) << "graph not captured";

  hidden_states_.copy_(hidden_states, true);
  encoder_hidden_states_.copy_(encoder_hidden_states, true);

  pooled_projections_.copy_(pooled_projections, true);
  timestep_.copy_(timestep, true);

  img_ids_.copy_(img_ids, true);
  txt_ids_.copy_(txt_ids, true);

  guidance_.copy_(guidance, true);

  // replay the graph
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  aclError st = aclmdlRIExecuteAsync(model_, stream);

  CHECK_EQ(st, ACL_SUCCESS)
      << "aclmdlRIExecuteAsync failed, error code: " << st;

  return output_;
}

}  // namespace xllm
