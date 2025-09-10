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

#include "dit_master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <csignal>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "models/model_registry.h"
#include "runtime/speculative_engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "server/xllm_server_registry.h"
#if defined(USE_NPU)
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#endif
#include "util/device_name_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
DITMaster::DITMaster(const Options& options)
    : Master(options, EngineType::DIT) {
  // TODO: init master
  InstanceInfo instance_info;
  XServiceClient::get_instance()->register_instance(instance_info);
  threadpool_ = std::make_unique<ThreadPool>(options_.num_handling_threads());
}

DITMaster::~DITMaster() {
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }

  // torch::cuda::empty_cache();
#if defined(USE_NPU)
  c10_npu::NPUCachingAllocator::emptyCache();
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu empty cache
#endif
}

void DITMaster::handle_batch_request(std::vector<ImageRequestParams> sps,
                                     BatchImageOutputCallback callback) {
  const size_t num_requests = sps.size();
  scheduler_->incr_pending_requests(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(sps[i]),
                   std::nullopt,
                   [i, callback](const ImageRequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void DITMaster::handle_request(ImageRequestParams sp,
                               std::optional<Call*> call,
                               ImageOutputCallback callback) {
  auto cb = [callback = std::move(callback)](const ImageRequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };
  LOG(INFO) << "in MM_master.cpp, into handle_request with prompt";
  // add into the queue
  threadpool_->schedule(
      [this, sp = std::move(sp), callback = std::move(cb), call]() mutable {
        AUTO_COUNTER(request_handling_latency_seconds_completion);

        // remove the pending request after scheduling
        SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

        Timer timer;
        // verify the prompt
        if (!sp.verify_params(callback)) {
          return;
        }

        auto request = generate_request(sp, call, callback);
        if (!request) {
          return;
        }

        if (!scheduler_->add_request(request)) {
          CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                              "No available resources to schedule request");
        }
      });
}

void DITMaster::run() {
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "DITMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      // scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

void DITMaster::generate() {
  DCHECK(options_.enable_schedule_overlap())
      << "Mode generate does not support schedule overlap yet.";
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "Generate is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  scheduler_->generate();
  running_.store(false, std::memory_order_relaxed);
}

std::shared_ptr<Request> DITMaster::generate_request(
    ImageRequestParams sp,
    std::optional<Call*> call,
    ImageOutputCallback callback) {
  DITRequestState req_state(std::move(sp.input_params),
                            std::move(sp.generation_params));

  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id,
                                           sp.offline,
                                           sp.slo_ms,
                                           sp.priority);

  // add one sequence, rest will be added by scheduler
  return request;
}

void DITMaster::get_cache_info(std::vector<uint64_t>& cluster_ids,
                               std::vector<std::string>& addrs,
                               std::vector<int64_t>& k_cache_ids,
                               std::vector<int64_t>& v_cache_ids) {
  engine_->get_cache_info(cluster_ids, addrs, k_cache_ids, v_cache_ids);
}

bool DITMaster::link_cluster(const std::vector<uint64_t>& cluster_ids,
                             const std::vector<std::string>& addrs,
                             const std::vector<std::string>& device_ips,
                             const std::vector<uint16_t>& ports,
                             const int32_t dp_size) {
  return engine_->link_cluster(cluster_ids, addrs, device_ips, ports, dp_size);
}

bool DITMaster::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                               const std::vector<std::string>& addrs,
                               const std::vector<std::string>& device_ips,
                               const std::vector<uint16_t>& ports,
                               const int32_t dp_size) {
  return engine_->unlink_cluster(
      cluster_ids, addrs, device_ips, ports, dp_size);
}

}  // namespace xllm