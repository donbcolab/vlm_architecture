# Best Practices when Building VLMs

- based on the arXiv paper [What matters when building vision-language models?](https://arxiv.org/pdf/2405.02246v1)

## Building Vision-Language Models: Idefics2

**Objective**: This paper explores the design and development of Idefics2, an 8-billion-parameter vision-language model (VLM) that achieves superior performance and efficiency on multiple benchmarks.

**Key Contributions**:
- Development of Idefics2 with architectural innovations and methodological optimizations.
- Performance evaluation against state-of-the-art VLMs, demonstrating significant improvements.

**Background**:
- Vision-language models are crucial for tasks that require understanding both visual and textual information.
- Challenges include balancing performance, efficiency, and training stability.

**Architectural Design**:
- **Overall Architecture**: Idefics2 combines a vision transformer and a powerful language model backbone.
- **Key Components**: Includes Idefics2VisionTransformer, MistralForCausalLM, and multi-modal projector.

**Methodological Choices**:
- **Cross-Attention**: Favored for better performance with frozen backbones.
- **Learned Pooling**: Reduces visual tokens, improving efficiency.
- **Aspect Ratio Preservation**: Maintains image quality and speeds up training.
- **LoRA**: Used for stable and efficient fine-tuning.

**Experimental Results**:
- **Benchmarking**: Idefics2 outperforms state-of-the-art models on various benchmarks.
- **Performance Gains**: Significant improvements in tasks like TextVQA and DocVQA.

**Key Findings and Best Practices**:
- **Finding 1**: Invest in high-quality language model backbones.
- **Finding 2**: Use cross-attention architectures for frozen backbones.
- **Finding 3**: Apply LoRA for stable training.
- **Finding 4**: Implement learned pooling for efficiency.
- **Finding 5**: Preserve aspect ratios in vision encoders.
- **Finding 6**: Split images into sub-images for better performance on text extraction tasks.

**Implications and Future Work**:
- **Implications**: Offers insights for improving VLMs, with applications in various domains.
- **Future Work**: Suggests exploring further optimizations and new architectures.

**Conclusion**:
- **Summary**: Idefics2 sets a new standard in VLMs with its innovative design and superior performance.
- **Takeaways**: Emphasizes best practices and methodological choices for future VLM development.

## Findings Summary

| Finding | Supporting Rationale | Impact | Architecture / Implementation |
| --- | --- | --- | --- |
| **Finding 1**: For a fixed number of parameters, the quality of the language model backbone has a higher impact on the performance of the final VLM than the quality of the vision backbone. | Experiments showed significant performance boosts when upgrading the language model backbone compared to the vision backbone. | Prioritizes investments in enhancing the language model backbone for better performance in VLMs. | **Implemented by**: Using powerful language models like `MistralForCausalLM` and `LlamaForCausalLM` across various models (e.g., `HuggingFaceM4/idefics2-8b-base`, `llava-hf/llava-1.5-7b-hf`). |
| **Finding 2**: The cross-attention architecture performs better than the fully autoregressive one when unimodal pre-trained backbones are kept frozen. | Cross-attention architecture achieved better scores with frozen backbones due to more frequent interleaving of vision and language model layers. | Suggests using cross-attention for scenarios with frozen backbones to maximize performance. | **Implemented by**: Using cross-attention mechanisms like `CLIPAttention` and `SiglipAttention` in models such as `TIGER-Lab/Mantis-8B-clip-llama3` and `TIGER-Lab/Mantis-8B-siglip-llama3`. |
| **Finding 3**: Unfreezing the pre-trained backbones under the fully autoregressive architecture can lead to training divergences. Leveraging LoRA still adds expressivity to the training and stabilizes it. | Fully autoregressive architecture showed instability when pre-trained backbones were unfrozen, but LoRA adaptation stabilized training. | Highlights the importance of using parameter-efficient fine-tuning methods like LoRA for stable training. | **Implemented by**: Although LoRA isn't explicitly mentioned in the table, the use of parameter-efficient fine-tuning methods is inferred from the model architecture and stability considerations. |
| **Finding 4**: Reducing the number of visual tokens with learned pooling significantly improves compute efficiency at training and inference while improving performance on downstream tasks. | Learned pooling reduced the number of visual tokens from 729 to 64, increasing performance by 8.5 points on average. | Demonstrates a method to balance computational efficiency and performance improvement. | **Implemented by**: Using learned pooling strategies in `Idefics2VisionTransformer` for `HuggingFaceM4/idefics2-8b-base`. |
| **Finding 5**: Adapting a vision encoder pre-trained on fixed-size square images to preserve images' original aspect ratio and resolution does not degrade performance while speeding up training and inference and reducing memory. | Aspect ratio-preserving strategy maintained performance levels and unlocked computational flexibility. | Enables efficient handling of images at various resolutions without compromising performance. | **Implemented by**: Using aspect ratio preservation strategies in vision transformers like `SiglipVisionTransformer` in `TIGER-Lab/Mantis-8B-siglip-llama3`. |
| **Finding 6**: Splitting images into sub-images during training allows trading compute efficiency for more performance during inference. The increase in performance is particularly noticeable in tasks involving reading text in an image. | Splitting images into sub-images boosted performance significantly on TextVQA and DocVQA benchmarks. | Provides a strategy to enhance performance in text extraction tasks from images by increasing compute during training. | **Implemented by**: Detailed processing of visual tokens in models like `Idefics2VisionTransformer` and `CLIPVisionTransformer`, supporting image splitting strategies. |
