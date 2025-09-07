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

#include "clip_tokenizer.h"

#include <absl/strings/escaping.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <absl/strings/string_view.h>
#include <glog/logging.h>
#include <re2/re2.h>

#include <fstream>
#include <optional>
#include <string>
#include <string_view>

#include "sentencepiece/util.h"

namespace xllm {

CLIPTokenizer::CLIPTokenizer(const std::string_view& dir_path,
                             const TokenizerArgs& args)
    : dir_path_(dir_path), args_(args) {
  // load vocab from file
  const std::string vocab_file_path =
      dir_path.empty() ? args.vocab_file()
                       : absl::StrCat(dir_path_, "/", args.vocab_file());
  load_vocab(vocab_file_path);
  // load merge file if exists
  const std::string merge_file_path =
      dir_path.empty() ? "merges.txt" : absl::StrCat(dir_path_, "/merges.txt");
  load_merges(merge_file_path);
  // add special tokens and construct special token regex
  if (!args.special_tokens().empty()) {
    const auto vocab_size = encoder_.size();
    load_special_tokens(args.special_tokens());
  }

  // construct regex
  if (!args.pattern().empty()) {
    const auto regex_str = absl::StrCat("(", args.pattern(), ")");
    regex_ = std::make_unique<re2::RE2>(regex_str);
    if (regex_->error_code() != 0) {
      LOG(FATAL) << "Failed to compile regex: " << args.pattern()
                 << ", error: " << regex_->error();
    }
  } else {
    const std::string clip_pattern =
        "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]"
        "+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";
    const auto regex_str = absl::StrCat("(", clip_pattern, ")");
    regex_ = std::make_unique<re2::RE2>(regex_str);
  }

  // construct prefix tokens
  if (!args.prefix_tokens().empty()) {
    for (const auto& token : args.prefix_tokens()) {
      if (token.empty()) {
        continue;
      }
      const auto token_id = token_to_id(token);
      if (token_id.has_value()) {
        prefix_token_ids_.push_back(token_id.value());
        LOG(INFO) << "Prefix token: " << token << ", id: " << token_id.value();
      } else {
        LOG(ERROR) << "Failed to find prefix token: " << token;
      }
    }
  }
}

void CLIPTokenizer::load_special_tokens(
    const std::vector<SpecialToken>& special_tokens) {
  // for each special token, add to encoder and decoder
  for (const auto& [token, id] : special_tokens) {
    if (token.empty()) {
      continue;
    }

    if (!special_token_encoder_.try_emplace(token, id).second) {
      LOG(WARNING) << "Duplicate special token: " << token << ", id: " << id;
    }

    if (!special_token_decoder_.try_emplace(id, token).second) {
      LOG(WARNING) << "Duplicate special token: " << token << ", id: " << id;
    }
  }

  // build special token regex
  std::vector<std::string> escaped_tokens;
  escaped_tokens.reserve(special_tokens.size());
  for (const auto& [token, id] : special_tokens) {
    if (token.empty()) {
      continue;
    }
    // escape each token
    const auto escaped_token = re2::RE2::QuoteMeta(token);
    escaped_tokens.push_back(escaped_token);
  }
  if (!escaped_tokens.empty()) {
    const auto special_token_regex_str = absl::StrJoin(escaped_tokens, "|");
    // surround with () to match special tokens
    const auto regex_str = absl::StrCat("(", special_token_regex_str, ")");
    special_token_regex_ = std::make_unique<re2::RE2>(regex_str);
  }
}

void CLIPTokenizer::load_vocab(const std::string& vocab_file_path) {
  // read token + rank from vocab file
  std::ifstream fs(vocab_file_path);
  if (!fs) {
    LOG(FATAL) << "Failed to open vocab file: " << vocab_file_path;
  }

  std::string line;
  while (std::getline(fs, line)) {
    if (line.empty()) {
      // skip empty line
      continue;
    }
    // split line by space
    const std::vector<std::string> parts = absl::StrSplit(line, ' ');
    if (parts.size() != 2) {
      LOG(WARNING) << "Failed to parse line: " << line;
      continue;
    }
    // parse token and rank
    std::string token;
    if (!absl::Base64Unescape(parts[0], &token)) {
      LOG(WARNING) << "Failed to parse token: " << parts[0];
      continue;
    }
    int32_t rank = 0;
    if (!absl::SimpleAtoi(parts[1], &rank)) {
      LOG(WARNING) << "Failed to parse rank: " << parts[1];
      continue;
    }

    if (!encoder_.try_emplace(token, rank).second) {
      LOG(WARNING) << "Duplicate token: " << token;
    }
    if (!decoder_.try_emplace(rank, token).second) {
      LOG(WARNING) << "Duplicate rank: " << rank;
    }
    vocab_set_.insert(token);
  }
}
void CLIPTokenizer::load_merges(const std::string& merges_file_path) {
  std::ifstream fs(merges_file_path);
  if (!fs) {
    LOG(FATAL) << "Failed to open merges file: " << merges_file_path;
  }

  std::string line;
  int32_t rank = 0;
  if (!std::getline(fs, line)) {
    LOG(FATAL) << "Empty merges file: " << merges_file_path;
  }

  while (std::getline(fs, line)) {
    if (line.empty()) {
      continue;
    }

    const std::vector<std::string> parts = absl::StrSplit(line, ' ');
    if (parts.size() != 2) {
      LOG(WARNING) << "Invalid merge line: " << line;
      continue;
    }

    bpe_ranks_[{parts[0], parts[1]}] = rank;
    rank++;
  }

  LOG(INFO) << "Loaded " << bpe_ranks_.size() << " BPE merges";
}

std::set<std::pair<std::string_view, std::string_view>>
CLIPTokenizer::get_pairs(const std::vector<std::string_view>& word) const {
  std::set<std::pair<std::string_view, std::string_view>> pairs;
  if (word.size() < 2) {
    return pairs;
  }
  for (size_t i = 0; i < word.size() - 1; ++i) {
    pairs.emplace(word[i], word[i + 1]);
  }
  return pairs;
}

std::vector<std::string_view> CLIPTokenizer::byte_pair_encode(
    const std::string_view& token) const {
  std::vector<std::string> word_strs;
  if (token.empty()) {
    word_strs.emplace_back("</w>");
  } else {
    for (size_t i = 0; i < token.size() - 1; ++i) {
      word_strs.emplace_back(1, token[i]);
    }
    word_strs.emplace_back(1, token.back());
    word_strs.back() += "</w>";
  }
  std::vector<std::string_view> word;
  word.reserve(word_strs.size());
  for (const auto& s : word_strs) {
    word.emplace_back(s);
  }

  auto pairs = get_pairs(word);
  if (pairs.empty()) {
    return word;
  }

  while (true) {
    std::pair<std::string_view, std::string_view> best_bigram;
    int32_t min_rank = std::numeric_limits<int32_t>::max();
    bool found_valid_bigram = false;

    for (const auto& bigram : pairs) {
      auto rank_it = bpe_ranks_.find(bigram);
      if (rank_it != bpe_ranks_.end() && rank_it->second < min_rank) {
        min_rank = rank_it->second;
        best_bigram = bigram;
        found_valid_bigram = true;
      }
    }
    if (!found_valid_bigram) {
      break;
    }

    const auto& first = best_bigram.first;
    const auto& second = best_bigram.second;
    std::vector<std::string> new_word_strs;
    size_t i = 0;
    while (i < word.size()) {
      size_t j = i;
      while (j < word.size() && word[j] != first) {
        j++;
      }
      if (j == word.size()) {
        for (; i < word.size(); ++i) {
          new_word_strs.emplace_back(word[i]);
        }
        break;
      }
      for (; i < j; ++i) {
        new_word_strs.emplace_back(word[i]);
      }
      if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
        new_word_strs.emplace_back(first);
        new_word_strs.back() += second;
        i += 2;
      } else {
        new_word_strs.emplace_back(word[i]);
        i += 1;
      }
    }
    word_strs.swap(new_word_strs);
    word.clear();
    word.reserve(word_strs.size());
    for (const auto& s : word_strs) {
      word.emplace_back(s);
    }
    if (word.size() == 1) {
      break;
    }
    pairs = get_pairs(word);
  }

  return word;
}
// void CLIPTokenizer::byte_pair_encode(const std::string_view& piece,
//                                          std::vector<int32_t>* ids) const {
//   if (piece.empty()) {
//     // empty piece, no need to encode
//     return;
//   }

//   // This is a vector of (start, rank) pairs.
//   // The rank is of the byte pair startig at position start.
//   // The rank of the last item in the vector is not a valid value.
//   std::vector<std::pair<int32_t, int32_t>> parts;
//   parts.reserve(piece.size() + 1);
//   const int32_t kMaxRank = std::numeric_limits<int32_t>::max();
//   for (int32_t i = 0; i <= piece.size(); ++i) {
//     parts.emplace_back(i, kMaxRank);
//   }

//   auto get_rank = [&piece, &parts, this](
//                       int32_t start, int32_t skip) -> std::optional<int32_t>
//                       {
//     if (start + skip + 2 < parts.size()) {
//       auto s = parts[start].first;
//       auto e = parts[start + skip + 2].first;
//       const auto key = piece.substr(s, e - s);
//       auto it = encoder_.find({key.data(), key.size()});
//       if (it != encoder_.end()) {
//         return it->second;
//       }
//     }
//     return std::nullopt;
//   };

//   // We look up the ranks once in the beginning and iteratively update
//   // them during each merge, which reduces the number of rank lookups.
//   for (int32_t i = 0; i < parts.size() - 2; ++i) {
//     const auto rank = get_rank(i, 0);
//     if (rank.has_value()) {
//       // kMaxRank is a sentinel value and cannot be a valid rank.
//       CHECK(rank.value() != kMaxRank) << "Invalid rank";
//       parts[i].second = rank.value();
//     }
//   }

//   while (parts.size() > 1) {
//     // find i with min rank. kMaxRank is a sentinel value.
//     int32_t min_rank = kMaxRank;
//     int32_t min_i = 0;
//     for (int32_t i = 0; i < parts.size() - 1; ++i) {
//       const auto rank = parts[i].second;
//       if (rank < min_rank) {
//         min_rank = rank;
//         min_i = i;
//       }
//     }

//     if (min_rank == kMaxRank) {
//       // No more merges possible.
//       break;
//     }

//     // remove parts[min_i + 1].
//     parts[min_i].second = get_rank(min_i, 1).value_or(kMaxRank);
//     if (min_i > 0) {
//       parts[min_i - 1].second = get_rank(min_i - 1, 1).value_or(kMaxRank);
//     }
//     parts.erase(parts.begin() + min_i + 1);
//   }

//   for (int32_t i = 0; i < parts.size() - 1; ++i) {
//     const auto s = parts[i].first;
//     const auto e = parts[i + 1].first;
//     // get rank for each piece
//     const auto key = piece.substr(s, e - s);
//     auto it = encoder_.find({key.data(), key.size()});
//     if (it == encoder_.end()) {
//       LOG(ERROR) << "Failed to find key: " << key;
//     } else {
//       ids->push_back(it->second);
//     }
//   }
// }

void CLIPTokenizer::encode_internal(const std::string_view& text,
                                    std::vector<int32_t>* ids) const {
  absl::string_view input{text.data(), text.size()};
  absl::string_view piece;
  // std::string_view piece;
  std::vector<std::string_view> bpe_ids;
  while (re2::RE2::FindAndConsume(&input, *regex_, &piece)) {
    auto it = encoder_.find(piece);
    if (it != encoder_.end()) {
      ids->push_back(it->second);
      continue;
    }
    // byte_pair_encode({piece.data(), piece.size()}, ids);
    bpe_ids = byte_pair_encode({piece.data(), piece.size()});
  }
  for (const auto& bpe_id : bpe_ids) {
    auto it = encoder_.find(bpe_id);
    if (it != encoder_.end()) {
      ids->push_back(it->second);
    } else {
      LOG(ERROR) << "Failed to find BPE id: " << bpe_id;
    }
  }
}

bool CLIPTokenizer::encode(const std::string_view& text,
                           std::vector<int32_t>* ids) const {
  // prepend prefix tokens if exists
  if (!prefix_token_ids_.empty()) {
    ids->insert(
        ids->begin(), prefix_token_ids_.begin(), prefix_token_ids_.end());
  }

  if (special_token_regex_ == nullptr) {
    encode_internal(text, ids);
    return true;
  }

  absl::string_view input{text.data(), text.size()};
  absl::string_view special;
  while (true) {
    const auto* start = input.begin();
    if (!re2::RE2::FindAndConsume(&input, *special_token_regex_, &special)) {
      // no more special tokens
      break;
    }

    // encode text before special token if exists
    const std::string_view sub_input(start,
                                     input.begin() - start - special.size());
    encode_internal(sub_input, ids);

    // add special token id if exists
    const auto sit = special_token_encoder_.find(special);
    if (sit != special_token_encoder_.end()) {
      // find one special token
      ids->push_back(sit->second);
    }
  }

  // encode remaining text if exists
  encode_internal({input.data(), input.size()}, ids);
  return true;
}

std::string CLIPTokenizer::decode(const Slice<int32_t>& ids,
                                  bool skip_special_tokens) const {
  std::stringstream ss;
  for (const auto& id : ids) {
    // encode special token
    const auto sit = special_token_decoder_.find(id);
    if (sit != special_token_decoder_.end()) {
      if (!skip_special_tokens) {
        ss << sit->second;
      }
      continue;
    }

    // encode token
    const auto it = decoder_.find(id);
    if (it != decoder_.end()) {
      ss << it->second;
      continue;
    }
    LOG(ERROR) << "Failed to find token for id: " << id;
  }

  std::string data = ss.str();
  absl::string_view bytes(data);

  // replace unfinished utf8 bytes with � (U+FFFD)
  std::stringstream utf8_ss;
  size_t offset = 0;
  while (offset < bytes.size()) {
    size_t consumed = 0;
    const bool is_valid = sentencepiece::string_util::IsValidDecodeUTF8(
        bytes.substr(offset), &consumed);
    if (is_valid) {
      utf8_ss << bytes.substr(offset, consumed);
    } else {
      // add replacement character � (U+FFFD) in UTF-8
      utf8_ss << "�";
      break;
    }
    offset += consumed;
  }
  return utf8_ss.str();
}

size_t CLIPTokenizer::vocab_size() const {
  // vocab size = encoder size + special tokens size
  return encoder_.size() + args_.special_tokens().size();
}

std::unique_ptr<Tokenizer> CLIPTokenizer::clone() const {
  return std::make_unique<CLIPTokenizer>(dir_path_, args_);
}

std::optional<int32_t> CLIPTokenizer::token_to_id(
    const std::string_view& token) const {
  const absl::string_view token_view{token.data(), token.size()};
  // encode special token
  const auto sit = special_token_encoder_.find(token_view);
  if (sit != special_token_encoder_.end()) {
    return sit->second;
  }

  // encode token
  const auto it = encoder_.find(token_view);
  if (it != encoder_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::string CLIPTokenizer::id_to_token(int32_t id) const {
  // encode special token
  const auto sit = special_token_decoder_.find(id);
  if (sit != special_token_decoder_.end()) {
    return sit->second;
  }

  // encode token
  const auto it = decoder_.find(id);
  if (it != decoder_.end()) {
    return it->second;
  }
  return "";
}

}  // namespace xllm
