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

#include <absl/container/flat_hash_map.h>
#include <re2/re2.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tokenizer.h"
#include "tokenizer_args.h"
namespace xllm {
struct PairHash {
  std::size_t operator()(
      const std::pair<std::string_view, std::string_view>& p) const noexcept {
    const std::hash<std::string_view> sv_hash;
    std::size_t h1 = sv_hash(p.first);
    std::size_t h2 = sv_hash(p.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};
class CLIPTokenizer : public Tokenizer {
 public:
  CLIPTokenizer(const std::string_view& dir_path, const TokenizerArgs& args);

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override;

  std::string id_to_token(int32_t id) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  void load_special_tokens(const std::vector<SpecialToken>& special_tokens);

  void load_vocab(const std::string& vocab_file_path);

  void load_merges(const std::string& merges_file_path);

  void encode_internal(const std::string_view& text,
                       std::vector<int32_t>* ids) const;

  std::set<std::pair<std::string_view, std::string_view>> get_pairs(
      const std::vector<std::string_view>& word) const;

  std::vector<std::string_view> byte_pair_encode(
      const std::string_view& token) const;

  std::string dir_path_;

  TokenizerArgs args_;

  // token to ids
  absl::flat_hash_map<std::string, int32_t> encoder_;
  // id to token
  absl::flat_hash_map<int32_t, std::string> decoder_;

  // a regex pattern to tokenize text
  // N.B. RE2 doesn't support look-around assertions.
  // https://github.com/google/re2/wiki/Syntax
  std::unique_ptr<re2::RE2> regex_;

  // special tokens to ids
  absl::flat_hash_map<std::string, int32_t> special_token_encoder_;

  // special token ids to tokens
  absl::flat_hash_map<int32_t, std::string> special_token_decoder_;

  // special token regex (optional)
  std::unique_ptr<re2::RE2> special_token_regex_;

  // token ids to add to the beginning of the input sequence
  std::vector<int32_t> prefix_token_ids_;

  // bpe regulations
  std::unordered_map<std::pair<std::string_view, std::string_view>,
                     int32_t,
                     PairHash>
      bpe_ranks_;
  std::unordered_set<std::string> vocab_set_;
};

}  // namespace xllm
