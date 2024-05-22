# Training and Evaluation - Processes and Data

### Summary Table: Training, Testing, and Evaluation Processes for Idefics2, Mantis, and LLaVA Models

| **Model**                         | **Training Data Sets**                                  | **Testing Data Sets**                                   | **Evaluation Process**                                                                                           | **Mention of EleutherAI Evaluation Process** |
|-----------------------------------|---------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| **Idefics2**                      | - Wikipedia, Public Multimodal Dataset, LAION, OBELICS  | - COCO Caption, TextVQA, NoCaps                        | - Benchmarked against state-of-the-art models on multiple datasets using standard metrics like BLEU, CIDEr, SPICE | No                                          |
| **Mantis**                        | - LAION-400M, CC3M, SBU                                  | - COCO Caption, Visual Genome, TextVQA, DocVQA         | - Evaluated on vision-language tasks using metrics such as BLEU, CIDEr, SPICE                                     | No                                          |
| **LLaVA**                         | - Common Crawl, COCO, Flickr30k, LAION, ImageNet, WIT   | - COCO Caption, Visual Genome, TextVQA, DocVQA         | - Evaluated using the LMMs-Eval pipeline, supporting multiple public datasets and new dataset onboarding          | Yes, based on lm-evaluation-harness          |
| **LLaVA-NeXT**                    | - Not specified in detail, likely similar datasets      | - COCO Caption, Visual Genome, TextVQA, DocVQA         | - Zero-shot modality transfer, DPO training with AI feedback on videos                                           | Yes, based on lm-evaluation-harness          |

### Key Findings and Evaluation Methodologies

- **Training Data Sets**: Extensive use of large-scale datasets such as LAION-400M, Common Crawl, COCO, and others, ensuring a wide variety of multimodal data.
- **Testing Data Sets**: Consistent use of standard datasets like COCO Caption, Visual Genome, TextVQA, and DocVQA to benchmark the models' performance.
- **Evaluation Processes**: 
  - Idefics2 and Mantis models are evaluated using common metrics (BLEU, CIDEr, SPICE) against state-of-the-art benchmarks.
  - LLaVA models use the LMMs-Eval pipeline, which supports evaluation on numerous public datasets and facilitates rapid development and testing of new models.
- **EleutherAI Evaluation Process**: The LMMs-Eval framework for LLaVA models is built upon EleutherAI’s lm-evaluation-harness, ensuring a robust and consistent evaluation methodology.

### Detailed Information

#### Idefics2
- **Training**: Leveraged large-scale datasets including Wikipedia, Public Multimodal Dataset, LAION, and a newly created dataset named OBELICS, which consists of 141 million interleaved image-text documents.
- **Testing and Evaluation**: Benchmarked on datasets like COCO Caption, TextVQA, and NoCaps. Evaluated using standard metrics such as BLEU, CIDEr, and SPICE【126†source】【129†source】.

#### Mantis
- **Training**: Utilized datasets like LAION-400M, CC3M, and SBU to train the model with a focus on multimodal data.
- **Testing and Evaluation**: Performance evaluated on vision-language tasks using COCO Caption, Visual Genome, TextVQA, and DocVQA datasets. Metrics used include BLEU, CIDEr, and SPICE【128†source】.

#### LLaVA (Large Language and Vision Assistant)
- **Training**: Employed datasets from Common Crawl, COCO, Flickr30k, LAION, ImageNet, and WIT.
- **Testing and Evaluation**: LMMs-Eval pipeline used for evaluation, supporting multiple datasets and enabling efficient benchmarking. Utilized metrics from lm-evaluation-harness for consistent evaluation  .

#### LLaVA-NeXT
- **Training**: Similar datasets as LLaVA, but specific details not fully provided. Likely includes extensive and diverse multimodal datasets.
- **Testing and Evaluation**: Includes zero-shot modality transfer capabilities and DPO training with AI feedback on videos. Evaluated using the LMMs-Eval pipeline, similar to LLaVA, based on the EleutherAI’s lm-evaluation-harness framework  .

### Conclusion

The models detailed above demonstrate a comprehensive approach to training, testing, and evaluation, leveraging extensive datasets and robust methodologies. The LMMs-Eval framework, based on EleutherAI’s lm-evaluation-harness, plays a critical role in ensuring consistent and efficient evaluation of these large multimodal models.

### References
- [Idefics2 Paper](https://arxiv.org/abs/2405.02246v1)
- [Mantis Paper](https://arxiv.org/abs/2306.13394v4)
- [LLaVA GitHub Page](https://github.com/haotian-liu/LLaVA)
- [LMMs-Eval Blog](https://lmms-lab.github.io/lmms-eval-blog/lmms-eval-0.1/)
- [EleutherAI’s lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
