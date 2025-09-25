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

#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/util/slice.h"

namespace xllm {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual bool encode(const std::string_view& text,
                      std::vector<int32_t>* ids) const = 0;

  virtual bool batch_encode(const std::vector<std::string>& texts,
                            std::vector<std::vector<int32_t>>* ids) const {
    for (const auto& text : texts) {
      std::vector<int32_t> single_ids;
      if (!encode(text, &single_ids)) {
        return false;
      }
      ids->push_back(single_ids);
    }
    return true;
  }

  virtual std::string decode(const Slice<int32_t>& ids,
                             bool skip_special_tokens) const = 0;

  virtual std::optional<int32_t> token_to_id(
      const std::string_view& token) const = 0;

  virtual std::string id_to_token(int32_t id) const = 0;

  virtual size_t vocab_size() const = 0;

  virtual std::unique_ptr<Tokenizer> clone() const = 0;
};

}  // namespace xllm
