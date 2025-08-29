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

#include "prefix_cache_hash_murmur3.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <string.h>

#include <iostream>
#include <thread>

#include "common/metrics.h"
#include "util/hash_util.h"

namespace xllm {

PrefixCacheHashMurmur3::PrefixCacheHashMurmur3(uint32_t block_size,
                                               bool enable_service_routing)
    : PrefixCacheHash(block_size),
      enable_service_routing_(enable_service_routing),
      hash_value_len_(MURMUR_HASH3_VALUE_LEN) {
  if (enable_service_routing_) {
    db_kvcache_events_.set_front_value(new KvCacheEvent());
    db_kvcache_events_.set_back_value(new KvCacheEvent());
  }
}

PrefixCacheHashMurmur3::~PrefixCacheHashMurmur3() {
  LOG(INFO) << "block matched rate: " << block_match_rate();
  auto back = db_kvcache_events_.get_back_value();
  if (back) {
    delete back;
  }

  auto front = db_kvcache_events_.get_back_value();
  if (front) {
    delete front;
  }
}

std::vector<Block> PrefixCacheHashMurmur3::match(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks) {
  // allign tokens to block boundary
  const size_t n_tokens = round_down(token_ids.size(), block_size_);
  if (n_tokens == 0) {
    return std::vector<Block>();
  }

  const int64_t now = absl::ToUnixMicros(absl::Now());
  size_t n_blocks = n_tokens / block_size_;
  total_blocks_.fetch_add(n_blocks);

  auto tokens_slice = token_ids.slice(0, n_tokens);

  std::vector<Block> blocks;
  blocks.reserve(n_blocks);
  blocks.insert(
      blocks.end(), existed_shared_blocks.begin(), existed_shared_blocks.end());

  DNodeList node_list;

  size_t start_index = existed_shared_blocks.size() * block_size_;
  Murmur3Key murmur3_key =
      existed_shared_blocks.empty()
          ? Murmur3Key{}
          : Murmur3Key{existed_shared_blocks.back().get_immutable_hash_value()};
  for (size_t i = start_index; i < n_tokens; i += block_size_) {
    if (i == 0) {
      murmur_hash3(
          nullptr, token_ids.slice(i, i + block_size_), murmur3_key.data);
    } else {
      murmur_hash3(murmur3_key.data,
                   token_ids.slice(i, i + block_size_),
                   murmur3_key.data);
    }

    auto iter = murmur3_cached_blocks_.find(murmur3_key);
    if (iter != murmur3_cached_blocks_.end()) {
      blocks.push_back(iter->second->block);
      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      break;
    }
  }

  // update LRU list
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  matched_blocks_.fetch_add(blocks.size());

  int64_t int_rate_percent = static_cast<int64_t>(
      static_cast<double>(blocks.size()) * 100.0 / n_blocks);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_rate, int_rate_percent);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_num, blocks.size());

  return blocks;
}

size_t PrefixCacheHashMurmur3::insert(const Slice<int32_t>& token_ids,
                                      const Slice<Block>& blocks) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // allign tokens to block boundary
  const size_t n_blocks =
      std::min(token_ids.size() / block_size_, blocks.size());
  const size_t n_tokens = n_blocks * block_size_;

  if (n_blocks == 0) {
    return 0;
  }

  // truncate the token ids and blocks to boundary
  auto tokens_slice = token_ids.slice(0, n_tokens);
  auto blocks_slice = blocks.slice(0, n_blocks);

  DNodeList node_list;
  Murmur3Key murmur3_key;

  uint32_t block_idx = 0;
  std::vector<Murmur3Key> insert_list;
  insert_list.reserve(n_blocks);
  for (size_t i = 0; i < n_tokens; i += block_size_) {
    if (i == 0) {
      murmur_hash3(
          nullptr, tokens_slice.slice(i, i + block_size_), murmur3_key.data);
    } else {
      murmur_hash3(murmur3_key.data,
                   tokens_slice.slice(i, i + block_size_),
                   murmur3_key.data);
    }

    auto iter = murmur3_cached_blocks_.find(murmur3_key);
    if (iter != murmur3_cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      Node* new_node = new Node();

      new_node->block = blocks[block_idx];
      new_node->block.set_hash_value(murmur3_key.data, hash_value_len_);
      new_node->block.set_token_ids(token_ids.slice(i, i + block_size_));
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      murmur3_cached_blocks_.emplace(std::make_pair(murmur3_key, new_node));

      if (enable_service_routing_) {
        insert_list.emplace_back(murmur3_key);
      }

      num_blocks_++;
    }

    ++block_idx;
  }

  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }
  if (enable_service_routing_) {
    threadpool_.schedule([insert_list = std::move(insert_list), this]() {
      auto front_ptr = this->db_kvcache_events_.get_front_value();
      if (!front_ptr) {
        LOG(INFO) << "Front DoubleBufferKvCacheEvent is nullptr!";
        return;
      }
      if (!this->exited_.load()) {
        for (const auto& hash_id : insert_list) {
          front_ptr->removed_cache.erase(hash_id);
          front_ptr->stored_cache.insert(hash_id);
        }
      }
    });
  }

  return n_tokens;
}

size_t PrefixCacheHashMurmur3::evict(size_t n_blocks) {
  if (num_blocks_ == 0 || lru_lst_.is_empty()) {
    return 0;
  }

  size_t evict_count = 0;
  Node* iter_node = lru_lst_.get_first();
  std::vector<Murmur3Key> del_list;
  del_list.reserve(n_blocks);
  for (size_t i = 0; i < n_blocks;) {
    if (lru_lst_.is_last(iter_node)) {
      break;
    }

    if (iter_node->block.is_shared()) {  // in use
      iter_node = iter_node->next;

      continue;
    }

    Node* del_node = iter_node;

    iter_node = lru_lst_.remove_node(del_node);

    Murmur3Key token_hash_key(del_node->block.get_immutable_hash_value());

    if (enable_service_routing_) {
      del_list.emplace_back(token_hash_key.data);
    }

    murmur3_cached_blocks_.erase(token_hash_key);

    delete del_node;
    ++evict_count;
    --num_blocks_;
    ++i;
  }
  if (enable_service_routing_) {
    threadpool_.schedule([del_list = std::move(del_list), this]() {
      auto front_ptr = this->db_kvcache_events_.get_front_value();
      if (!front_ptr) {
        LOG(INFO) << "Front DoubleBufferKvCacheEvent is nullptr!";
        return;
      }
      if (!this->exited_.load()) {
        for (const auto& hash_id : del_list) {
          front_ptr->removed_cache.insert(hash_id);
          front_ptr->stored_cache.erase(hash_id);
        }
      }
    });
  }

  return evict_count;
}

KvCacheEvent* PrefixCacheHashMurmur3::get_upload_kvcache_events() {
  if (!enable_service_routing_) {
    return nullptr;
  }

  db_kvcache_events_.swap();
  if (!exited_.load()) {
    return db_kvcache_events_.get_back_value();
  } else {
    return nullptr;
  }
}
}  // namespace xllm
