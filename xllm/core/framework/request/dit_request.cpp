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

#include "dit_request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "api_service/call.h"

namespace xllm {

DITRequest::DITRequest(const std::string& request_id,
                       const std::string& x_request_id,
                       const std::string& x_request_time,
                       const DITRequestState& state,
                       const std::string& service_request_id)
    : RequestBase(request_id, x_request_id, x_request_time, service_request_id),
      state_(std::move(state)) {}

}  // namespace xllm
