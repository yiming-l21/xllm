/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_engine.h"

#include <glog/logging.h>
#include <sys/sysinfo.h>

#include "core/common/metrics.h"
#include "util/timer.h"
#include "worker.h"

namespace xllm {
DiTEngine::DiTEngine(const runtime::Options& options) : options_(options) {
  const auto& devices = options_.devices();
  LOG(INFO) << "Devices: " << devices;
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";
    int currentDevId = device.index();
#if defined(USE_NPU)
    int ret = aclrtSetDevice(currentDevId);
    if (ret != 0) {
      LOG(ERROR) << "ACL set device id:" << currentDevId
                 << " failed, ret:" << ret;
    }
#endif
  }

  if (devices.size() > 1) {
    // create a process group for each device if there are multiple gpus
    process_groups_ = ProcessGroup::create_process_groups(devices);
  }
  const int32_t world_size = static_cast<int32_t>(devices.size());

  // create workers
  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i);
    ProcessGroup* pg = world_size > 1 ? process_groups_[i].get() : nullptr;
    ParallelArgs parallel_args(rank, world_size, pg);
    workers_.emplace_back(
        std::make_unique<DiTWorker>(parallel_args, devices[i], options_));
  }

  if (workers_.size() > 1) {
    // test process group
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(workers_.size());
    for (auto& worker : workers_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    // wait up to 10 seconds for all futures to complete
    folly::collectAll(futures).within(std::chrono::seconds(10)).get();
  }
  LOG(INFO) << "DiT Engine Initialized done Using devices: " << devices;
}

bool DiTEngine::init() {
  if (!init_model()) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }
  return true;
}

bool DiTEngine::init_model() {
  const std::string& model_path = options_.model_path();
  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  LOG(INFO) << "Starting to init model on " << workers_.size() << " workers.";
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->init_model_async(model_path));
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  LOG(INFO) << "All workers completed model initialization.";
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }

  LOG(INFO) << "All workers successfully initialized the model.";
  return true;
}

DiTForwardOutput DiTEngine::step(std::vector<DiTBatch>& batches) {
  CHECK(!workers_.empty());

  Timer timer;
  auto forward_inputs = workers_[0]->prepare_inputs(batches[0]);
  COUNTER_ADD(prepare_input_latency_seconds, timer.elapsed_seconds());

  std::vector<folly::SemiFuture<std::optional<DiTForwardOutput>>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.emplace_back(worker->step_async(forward_inputs));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  // return the result from the driver
  auto forward_output = results.front().value();
  DCHECK(forward_output.has_value()) << "Failed to execute model";
  batches[0].process_forward_output(forward_output.value());
  return forward_output.value();
}

std::vector<int64_t> DiTEngine::get_active_activation_memory() const {
  // call worker to get active activation memory
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->get_active_activation_memory());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(workers_.size());
  for (auto& result : results) {
    active_activation_memories.push_back(result.value());
  }
  return active_activation_memories;
}
}  // namespace xllm
