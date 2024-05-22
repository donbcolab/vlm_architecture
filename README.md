# vlm_architecture

## VLM Comparison

- only a sample focusing on modes with less than 10 billion parameters

| Model                       | # of Parameters | Architecture               | Vision Encoder                        | Pooling          | Modality Projection       | Language Model Backbone        | Fine-tuning Method           | Flash Attention | Hugging Face Model Card URL                                                                                     |
|-----------------------------|-----------------|----------------------------|---------------------------------------|------------------|---------------------------|-------------------------------|------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------|
| **Idefics2-8b-base**        | 8B              | Fully autoregressive       | SigLIP-SO400M                         | Learned pooling  | Modality projection layer | Mistral-7B                    | LoRA (Low-Rank Adaptation)   | No              | [Idefics2-8b-base](https://huggingface.co/HuggingFaceM4/idefics2-8b-base)                                     |
| **Idefics2-8b**             | 8B              | Fully autoregressive       | SigLIP-SO400M                         | Learned pooling  | Modality projection layer | Mistral-7B                    | LoRA (Low-Rank Adaptation)   | No              | [Idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b)                                               |
| [**Idefics2-8b-chatty**](notebooks/hf_idefics2_chatty.ipynb)      | 8B              | Fully autoregressive       | SigLIP-SO400M                         | Learned pooling  | Modality projection layer | Mistral-7B                    | LoRA (Low-Rank Adaptation)   | No              | [Idefics2-8b-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty)                                  |
| [**LLaVA-v1.6-mistral-7B**](notebooks/llava_1_6_mistral.ipynb)   | 7B              | Auto-regressive            | openai/clip-vit-large-patch14-336     | Not specified    | Text-image interleaving   | Mistral-7B                    | Multimodal instruction data  | No              | [LLaVA-v1.6-mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)                               |
| **LLaVA-v1.6-vicuna-7B**    | 7B              | Auto-regressive            | openai/clip-vit-large-patch14-336     | Not specified    | Text-image interleaving   | Vicuna-7B                     | Multimodal instruction data  | No              | [LLaVA-v1.6-vicuna-7b](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)                                 |
| [**Mantis-8B-clip-llama3**](notebooks/mantis_8B_clip_llama3.ipynb)   | 8B              | Sequence-based             | openai/clip-vit-large-patch14-336     | Not specified    | Text-image interleaving   | Meta-Llama-3-8B-Instruct      | Instruction Tuning           | Optional              | [Mantis-8B-clip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-clip-llama3)                                |
| [**Mantis-8B-siglip-llama3**](notebooks/mantis_8B_siglip_llama3.ipynb) | 8B              | Sequence-based             | siglip_vision_model                   | Not specified    | Text-image interleaving   | Meta-Llama-3-8B-Instruct      | Instruction Tuning           | Optional              | [Mantis-8B-siglip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3)                            |
| [**Phi-3-vision-128k-instruct**](notebooks/ms_phi3_vision.ipynb) | 4.2B         | Hybrid auto-regressive     | custom-vision-transformer-128k       | Not specified    | Flash Attention v2        | Custom LLM                   | Hybrid instruction tuning    | Yes             | [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)                      |

