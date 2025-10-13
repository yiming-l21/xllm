/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "parallel_state.h"

#include <c10/core/Device.h>
#if defined(USE_NPU)
#include <hccl/hccl_types.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "hccl/hccl.h"
#endif
#pragma GCC diagnostic pop
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <vector>

#include "core/framework/model/model_args.h"

namespace xllm {

namespace {
#if defined(USE_NPU)
#define HCCLCHECK(cmd)                                               \
  do {                                                               \
    HcclResult r = cmd;                                              \
    if (r != HCCL_SUCCESS) {                                         \
      LOG(FATAL) << "Failed, HCCL error :" ; \
    }                                                                \
  } while (0)
#endif
inline bool is_npu(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().is_privateuseone();
}
inline bool is_npu(const at::TensorOptions& options) {
  return options.device().is_privateuseone();
}
inline bool is_npu(const at::Device& device) {
  return device.is_privateuseone();
}
at::Tensor flatten_for_scatter_gather(std::vector<at::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}
#if defined(USE_NPU)
HcclDataType to_hccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
  switch (type) {
    case at::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case at::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case at::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case at::kLong:
      return HCCL_DATA_TYPE_INT64;
    case at::kInt:
      return HCCL_DATA_TYPE_INT32;
    case at::kChar:
      return HCCL_DATA_TYPE_INT8;
    case at::kByte:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBool:
      return HCCL_DATA_TYPE_UINT8;
    case at::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    default:
      TORCH_CHECK(false, "Unconvertible HCCL type ", type);
  }
}
#endif
void check_input(torch::Tensor input) {
  CHECK(is_npu(input)) << "input should be npu tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be npu dense tensor";
}
}  // namespace

namespace parallel_state {

std::optional<ParallelArgs> get_dp_attn_parallel_args(
    const ParallelArgs& parallel_args) {
  if (parallel_args.dp_size() <= 1) {
    return std::nullopt;
  }

  // tp=1 in each dp group
  if (parallel_args.dp_size() == parallel_args.world_size()) {
    return ParallelArgs(0,  // local rank
                        1,  // world_size
                        nullptr,
                        nullptr,
                        parallel_args.dp_size());
  }

  return ParallelArgs(parallel_args.dp_local_process_group_->rank(),
                      parallel_args.dp_local_process_group_->world_size(),
                      parallel_args.dp_local_process_group_,
                      nullptr,
                      parallel_args.dp_size());
}

torch::Tensor gather(torch::Tensor input, const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  const auto rank = parallel_args.rank();
  // auto* process_group = parallel_args.process_group();
  std::vector<torch::Tensor> tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    tensors[i] = torch::empty_like(input);
  }
  // blocking call
  // process_group->allgather(input, tensors);
  return torch::cat(tensors, /*dim=*/-1).contiguous();
}

torch::Tensor reduce(torch::Tensor input, const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }
  // auto* process_group = parallel_args.process_group();
  // process_group->allreduce(input);
  return input;
}

torch::Tensor scatter(torch::Tensor input, const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  // get the size for last dimension
  const auto last_dim_size = input.size(-1);
  CHECK(last_dim_size % world_size == 0)
      << "last_dim_size " << last_dim_size << " not divisible by world_size "
      << world_size;

  // torch::split does not create contiguous tensors by default.
  const auto tensor_list = input.split(last_dim_size / world_size, /*dim=*/-1);
  const auto rank = parallel_args.rank();
  return tensor_list[rank];
}

std::vector<torch::Tensor> all_gather(torch::Tensor& input,
                                      const ParallelArgs& parallel_args) {
  const int world_size = parallel_args.world_size();
  if (world_size <= 1) {
    return {input};
  }
  auto* pg = parallel_args.process_group_;
  CHECK(pg != nullptr) << "all_gather: process_group_ is null";

  std::vector<torch::Tensor> outputs(world_size);
  for (int i = 0; i < world_size; ++i) {
    outputs[i] = torch::empty_like(input);
  }
  pg->allgather(input, outputs);
  return outputs;
}

torch::Tensor all_to_all_equal(torch::Tensor& send,
                               bool is_sync,
                               const ParallelArgs& parallel_args
#if defined(USE_NPU)
                               ,
                               std::shared_ptr<c10_npu::NPUEvent>* out_done
#endif
) {
  const int P = parallel_args.world_size();
  if (P <= 1) return send;
  auto* pg = parallel_args.process_group_;
  CHECK(pg != nullptr) << "all_to_all_equal: process_group_ is null";
  auto recv = torch::empty_like(send);
#if defined(USE_NPU)
  static_cast<ProcessGroupHCCL*>(pg)->alltoall_equal(
      send, recv, is_sync, out_done);
#else
  LOG(FATAL) << "all_to_all_equal only implemented for NPU";
#endif
  return recv;
}

}  // namespace parallel_state

#if defined(USE_NPU)
std::vector<std::unique_ptr<ProcessGroup>> ProcessGroup::create_process_groups(
    const std::vector<torch::Device>& devices) {
  CHECK(!devices.empty()) << "devices should not be empty";
  for (const auto& device : devices) {
    CHECK(is_npu(device)) << "device should be npu device";
  }
  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }
  std::vector<HcclComm> comms(devices.size());
  const int world_size = static_cast<int>(devices.size());
  HCCLCHECK(HcclCommInitAll(world_size, device_idxs.data(), comms.data()));
  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());
  for (int i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupHCCL>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }
  return process_groups;
}
#elif defined(USE_MLU)
// TODO(mlu): implement create_process_groups for mlu
std::vector<std::unique_ptr<ProcessGroup>> ProcessGroup::create_process_groups(
    const std::vector<torch::Device>& devices) {
  return { LOG(FATAL) << "Not implemented for mlu"; };
}
#endif

#if defined(USE_NPU)
ProcessGroupHCCL::ProcessGroupHCCL(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   HcclComm comm)
    : ProcessGroup(rank, world_size, device),
      comm_(comm),
      comm_stream_(c10_npu::getNPUStreamFromPool(device.index())) {}
// Destructor.
ProcessGroupHCCL::~ProcessGroupHCCL() { HCCLCHECK(HcclCommDestroy(comm_)); }

void ProcessGroupHCCL::allreduce(torch::Tensor& input) {
  DCHECK(input.device() == device())
      << "input should be on the same device as the process group";
  check_input(input);
}

void ProcessGroupHCCL::allgather(torch::Tensor input,
                                 std::vector<torch::Tensor>& outputs) {
  check_input(input);
  CHECK(outputs.size() == world_size())
      << "outputs should have the same size as world_size";
  DCHECK(input.device() == device())
      << "input should be on the same device as the process group";
  torch::DeviceGuard device_guard(device());
  torch::Tensor flattened_output = flatten_for_scatter_gather(outputs);
  const auto count = input.numel();
  const auto data_type = to_hccl_data_type(input);

  auto stream = c10_npu::getCurrentNPUStream();
  HCCLCHECK(HcclAllGather(
      /*sendbuff=*/input.data_ptr(),
      /*recvbuff=*/flattened_output.data_ptr(),
      /*sendcount=*/count,
      /*datatype=*/data_type,
      /*comm=*/comm_,
      /*stream=*/stream));
  // copy the flattened output tensors to the outputs.
  for (int i = 0; i < static_cast<int>(outputs.size()); ++i) {
    outputs[i].copy_(flattened_output[i], /*non_blocking=*/true);
  }
}

void ProcessGroupHCCL::alltoall_single(
    torch::Tensor send,
    torch::Tensor recv,
    const std::vector<int64_t>& send_splits,
    const std::vector<int64_t>& recv_splits,
    bool is_sync,
    std::shared_ptr<c10_npu::NPUEvent>* out_done) {
#if !defined(USE_NPU)
  LOG(FATAL) << "alltoall_single only supported with USE_NPU";
#else
  check_input(send);
  check_input(recv);
  CHECK(send.device() == device() && recv.device() == device())
      << "send/recv must be on the same device as the process group";
  const int P = world_size();
  CHECK((int)send_splits.size() == P && (int)recv_splits.size() == P)
      << "split sizes length must equal world_size";

  std::vector<uint64_t> sc(P), rc(P), sdisp(P), rdisp(P);
  uint64_t acc = 0;
  for (int i = 0; i < P; ++i) {
    sc[i] = static_cast<uint64_t>(send_splits[i]);
    sdisp[i] = acc;
    acc += sc[i];
  }
  acc = 0;
  for (int i = 0; i < P; ++i) {
    rc[i] = static_cast<uint64_t>(recv_splits[i]);
    rdisp[i] = acc;
    acc += rc[i];
  }

  auto dtype = to_hccl_data_type(send);
  auto compute_stream = c10_npu::getCurrentNPUStream();
  c10_npu::NPUEvent ready;
  ready.record(compute_stream);

  torch::DeviceGuard guard(device());
  // const auto prev_stream = c10_npu::getCurrentNPUStream();
  // c10_npu::setCurrentNPUStream(comm_stream_);
  ready.block(comm_stream_);  // compute -> comm
  c10_npu::NPUCachingAllocator::recordStream(send.storage().data_ptr(),
                                             comm_stream_);
  c10_npu::NPUCachingAllocator::recordStream(recv.storage().data_ptr(),
                                             comm_stream_);
  HCCLCHECK(HcclAlltoAllV(
      /*sendBuf=*/send.data_ptr(),
      /*sendCounts=*/sc.data(),
      /*sdispls=*/sdisp.data(),
      /*sendType=*/dtype,
      /*recvBuf=*/recv.data_ptr(),
      /*recvCounts=*/rc.data(),
      /*rdispls=*/rdisp.data(),
      /*recvType=*/dtype,
      /*comm=*/comm_,
      /*stream=*/comm_stream_.stream()));

  if (is_sync) {
    c10_npu::NPUEvent ev;
    ev.record(comm_stream_);
    ev.synchronize();
  } else {
    auto done = std::make_shared<c10_npu::NPUEvent>();
    done->record(comm_stream_);

    if (out_done) {
      *out_done = std::move(done);
    } else {
      done->block(compute_stream);
    }
  }
  // c10_npu::setCurrentNPUStream(prev_stream);
#endif
}
void ProcessGroupHCCL::flush_comm_to_current() {
#if defined(USE_NPU)
  auto cur = c10_npu::getCurrentNPUStream();
  c10_npu::NPUEvent fence;
  fence.record(comm_stream_);  // 通信流 -> 事件
  fence.block(cur);            // 事件 -> 当前计算流
#endif
}

void ProcessGroupHCCL::alltoall_equal(
    torch::Tensor send,
    torch::Tensor recv,
    bool is_sync,
    std::shared_ptr<c10_npu::NPUEvent>* out_done) {
#if !defined(USE_NPU)
  LOG(FATAL) << "alltoall_equal only supported with USE_NPU";
#else
  check_input(send);
  check_input(recv);
  const int P = world_size();
  CHECK(send.numel() % P == 0 && recv.numel() % P == 0)
      << "send/recv numel must be divisible by world_size";
  const int64_t per_rank_send = send.numel() / P;
  const int64_t per_rank_recv = recv.numel() / P;
  std::vector<int64_t> in_splits(P, per_rank_send);
  std::vector<int64_t> out_splits(P, per_rank_recv);
  alltoall_single(send, recv, in_splits, out_splits, is_sync, out_done);
#endif
}

#endif  // defined(USE_NPU)
}  // namespace xllm
