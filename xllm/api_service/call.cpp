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

#include "call.h"

namespace xllm {

Call::Call(brpc::Controller* controller) : controller_(controller) { init(); }

void Call::init() {
  if (controller_->http_request().GetHeader("x-request-id")) {
    x_request_id_ = *controller_->http_request().GetHeader("x-request-id");
  } else if (controller_->http_request().GetHeader("x-ms-client-request-id")) {
    x_request_id_ =
        *controller_->http_request().GetHeader("x-ms-client-request-id");
  }

  if (controller_->http_request().GetHeader("x-request-time")) {
    x_request_time_ = *controller_->http_request().GetHeader("x-request-time");
  } else if (controller_->http_request().GetHeader("x-request-timems")) {
    x_request_time_ =
        *controller_->http_request().GetHeader("x-request-timems");
  }
}

}  // namespace xllm
