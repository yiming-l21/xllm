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

#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "common/metrics.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/engine.h"
#include "scheduler/decode_priority_queue.h"
#include "util/utils.h"

namespace xllm {

namespace {
constexpr size_t kRequestQueueSize = 100;
}  // namespace

DiTDynamicBatchScheduler::DiTDynamicBatchScheduler(Engine* engine, const Options& options)
    : options_(options),
      engine_(engine),
      request_queue_(kRequestQueueSize) {
  CHECK(engine_ != nullptr);
}

DiTDynamicBatchScheduler::~DiTDynamicBatchScheduler() { running_requests_.clear(); }

bool DiTDynamicBatchScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);

  if (request_queue_.write(request)) {
    return true;
  }

  return false;
}


std::vector<Batch> DiTDynamicBatchScheduler::prepare_batch() {
  Timer timer;

  int count = 0;
  std::shared_ptr<Request> request;
  while (request_queue_.read(request)) {

    running_requests_.emplace_back(request);

    if(++count == options_.max_request_per_batch)
      break;
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            waiting_priority_queue_.size() + running_queue_->size());

  return batches;
}

std::vector<Batch> DiTDynamicBatchScheduler::schedule_request(
    const absl::Duration& timeout) {

  const auto deadline = absl::Now() + timeout;
  std::vector<Batch> batch;

  while (true) {
    batch = prepare_batch();
    bool all_empty =
        std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
          return one_batch.empty();
        });

    if (!all_empty) {
      return batch;
    }

    const auto now = absl::Now();
    if (now > deadline) {
      break;
    }
    // wait for new requests to arrive
    constexpr uint64_t kStepSleepTimeMs = 10;
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }

  // return an empty batch
  return batch;
}

void DiTDynamicBatchScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  std::vector<Batch> batch = schedule_request(timeout);
  bool all_empty =
      std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      });
  
  if (all_empty) {
    return;
  }

  engine_->step(batch);

  // process request output in batch
  process_batch_output(false);
}

void DiTDynamicBatchScheduler::process_batch_output(bool enable_schedule_overlap) {

}



}  // namespace xllm
