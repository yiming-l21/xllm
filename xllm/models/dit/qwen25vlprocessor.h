#include <string>

#include "core/framework/request/mm_data.h"
#include "models/vlm/qwen2_5_vl.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {
namespace qwenimage {

struct BatchFeature {
  vector<vector<int>> input_ids_;
  torch::Tensor attention_mask_;
  torch::Tensor pixel_values_;
  torch::Tensor image_frid_thw_;
  BatchFeature(vector<vector<int>>&& input_ids,
               torch::Tensor&& attention_mask,
               torch::Tensor&& pixel_values,
               torch::Tensor&& image_frid_thw)
      : input_ids_(std::move(input_ids)),
        attention_mask_(std::move(attention_mask)),
        pixel_values_(std::move(pixel_values)),
        image_frid_thw_(std::move(image_frid_thw)) {}
};

class Qwen2VLProcessor {
 public:
  Qwen2VLProcessor(std::unique_ptr<Qwen2VLImageProcessor> processor,
                   std::unique_ptr<Tokenizer> tokenizer,
                   const ModelArgs& args)
      : processor_(std::move(processor)),
        tokenizer_(std::move(tokenizer)),
        image_token_("<|image_pad|>"),
        video_token_("<|video_pad|>"),
        merge_length_(args.mm_image_merge_size() * args.mm_image_merge_size()) {
    image_token_id = tokenizer->token_to_id(image_token_).value_or(151655);
    video_token_id = tokenizer->token_to_id(video_token_).value_or(151656);
  }

  BatchFeature processor(vector<torch::Tensor>& images, vector<string>& text) {
    auto mmData = MMData();
    processor_->process_images(images, mmData);
    auto image_grid_thw = mmData.get<torch::Tensor>("image_grid_thw").value();
    auto pixel_values = mmData.get<torch::Tensor>("pixel_values").value();
    // std::vector<string> text_copy = text;
    vector<vector<int>> text_ids;
    for (auto prompt : text) {
      size_t index = 0;
      int64_t begin = 0;
      string data = "";

      auto end = prompt.find(image_token_, begin);

      while (end != std::string::npos) {
        data.append(prompt, begin, end - begin);
        auto token_num =
            image_grid_thw[index].prod().item<int>() / merge_length_;
        while (token_num--) data.append(image_token_);

        ++index;
        begin = end + image_token_.size();
        end = prompt.find(image_token_, begin);
      }

      if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);
      vector<int>* input_ids;
      tokenizer_->encode(data, input_ids);
      text_ids.emplace_back(*input_ids);
    }

    return BatchFeature(std::move(text_ids),
                        std::move(torch::Tensor()),
                        std::move(pixel_values),
                        std::move(image_grid_thw));
  }

 private:
  std::unique_ptr<Qwen2VLImageProcessor> processor_;
  std::unique_ptr<Tokenizer> tokenizer_;
  const string image_token_;
  const string video_token_;
  int64_t image_token_id;
  int64_t video_token_id;
  int64_t merge_length_;
};

}  // namespace qwenimage
}  // namespace xllm
