<div align="center">

<img src="assets/logo.png" width="30%"/>



<!-- <b> An auto-regressive generation model that combines discrete and continuous representations to process visual inputs, making it easy to integrate both visual understanding and image generation tasks. -->

<h3>UniToken: Harmonizing Multimodal Understanding and Generation through Unified Visual Encoding</h3>

[Yang Jiao](https://sxjyjay.github.io/)<sup>1,2</sup>, &nbsp; [Haibo Qiu](https://haibo-qiu.github.io/)<sup>3</sup>, &nbsp; [Zequn Jie](https://scholar.google.com/citations?user=4sKGNB0AAAAJ&hl=zh-CN&oi=sra)<sup>3</sup>, &nbsp; [Shaoxiang Chen](https://scholar.google.com/citations?user=WL5mbfEAAAAJ&hl=zh-CN)<sup>3</sup>, &nbsp; [Jingjing Chen](https://jingjing1.github.io/)<sup>1,2</sup>, &nbsp; </br>
[Lin Ma](https://forestlinma.com/)<sup>3</sup>, &nbsp; [Yu-Gang Jiang](https://fvl.fudan.edu.cn/)<sup>1,2</sup>

<sup>1</sup>Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University &nbsp; </br> 
<sup>2</sup>Shanghai Collaborative Innovation Center on Intelligent Visual Computing &nbsp; </br>
<sup>3</sup>Meituan 

[![UniToken](https://img.shields.io/badge/Paper-UniToken-d32f2f.svg?logo=arXiv)](https://arxiv.org/abs/2504.04423)&#160;
<a href='https://huggingface.co/OceanJay/UniToken-AnyRes-StageII'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-models-blue'></a><br>

</div>

<img src="assets/demo.png">

## 📣 News
- **[2025-04-02] 🎉🎉🎉 UniToken [paper](https://arxiv.org/abs/2504.04423) is accepted to CVPR 2025 workshop! 🎉🎉🎉**
- **[2025-04-01] 🎉🎉🎉 We release the [recaptioned text prompts](https://huggingface.co/datasets/OceanJay/rewrite_geneval_t2icompbench) of GenEval and T2I-Compbench! 🎉🎉🎉**
- **[2025-04-01] 🎉🎉🎉 UniToken [paper](https://arxiv.org/abs/2504.04423) and training codes are released! 🎉🎉🎉**

## 🛠️ Installation

See [INSTALL.md](./INSTALL.md) for detailed instructions.



## 🎓 Training
See [unitoken/TRAIN.md](unitoken/TRAIN.md)

## 🤖 Inference

<!-- > [!Note]
>
> Before using the Lumina-mGPT model, run
>
> ```bash
> # bash
> cd lumina_mgpt
> ```
>
> to enter the directory of the Lumina-mGPT implementation. -->

### Preparation

Download the original [VQ-VAE weights](https://github.com/facebookresearch/chameleon),  [Lumina-mGPT-512](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-512) and [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384), and put them to the following directory:

```
UniToken
- unitoken/
    - ckpts/
        - chameleon/
            - tokenizer/
                - text_tokenizer.json
                - vqgan.yaml
                - vqgan.ckpt
        - Lumina-mGPT-7B-512/
        - SigLIP/
- xllmx/
- ...
```



### Simple Inference

The simplest code for UniToken inference:

```python
from inference_solver_anyres import FlexARInferenceSolverAnyRes
from PIL import Image

# ******************** Image Generation ********************
inference_solver = FlexARInferenceSolverAnyRes(
    model_path="OceanJay/UniToken-AnyRes-StageII",
    precision="bf16",
    target_size=512,
)

q1 = f"Generate an image according to the following prompt:\n" \
     f"A majestic phoenix with fiery wings soaring above a tranquil mountain lake, casting shimmering reflections on the water. Sparks and embers trail behind it as the sky glows with hues of orange and gold."

# generated: tuple of (generated response, list of generated images)
generated = inference_solver.generate_img(
    images=[],
    qas=[[q1, None]],
    max_gen_len=1536,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=3.0, image_top_k=4000),
)

a1, new_image = generated[0], generated[1][0]


# ******************* Image Understanding ******************
inference_solver = FlexARInferenceSolverAnyRes(
    model_path="OceanJay/UniToken-AnyRes-StageII",
    precision="bf16",
    target_size=512,
)

# "<|image|>" symbol will be replaced with sequence of image tokens before fed to LLM
q1 = "<|image|>Please describe the details of the image as much as possible."

images = [Image.open("../assets/1.png").convert('RGB')]
qas = [[q1, None]]

# `len(images)` should be equal to the number of appearance of "<|image|>" in qas
generated = inference_solver.generate(
    images=images,
    qas=qas,
    max_gen_len=512,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
)

a1 = generated[0]
# generated[1], namely the list of newly generated images, should typically be empty in this case.
```

## 🤗 Checkpoints

| Model        |Huggingface                                                                              |
| ------------ | ---------------------------------------------------------------------------------------- |
| UniToken-base-StageI   | [OceanJay/UniToken-base-StageI](https://huggingface.co/OceanJay/UniToken-base-StageI)       |
| UniToken-base-StageII   | [OceanJay/UniToken-base-StageII](https://huggingface.co/OceanJay/UniToken-base-StageII)       |
| UniToken-AnyRes-StageI | [OceanJay/UniToken-AnyRes-StageI](https://huggingface.co/OceanJay/UniToken-AnyRes-StageI) |
| UniToken-AnyRes-StageII  | [OceanJay/UniToken-AnyRes-StageII](https://huggingface.co/OceanJay/UniToken-AnyRes-StageII)     |

## 📚 Datasets
We've observed that existing text-to-image generation models struggle with short text prompts in benchmarks such as GenEval and T2I-Compbench++. To address this issue, we have revised these prompts to be more descriptive. We are excited to share our enhanced version on [Hugging Face](https://huggingface.co/datasets/OceanJay/rewrite_geneval_t2icompbench). We encourage you to try it out and see the improvements for your own model!

## 🙏 Acknowledgement

We sincerely appreciate [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT) for providing high-quality training codes, as well as [Emu3](https://github.com/baaivision/Emu3) and [Janus](https://github.com/deepseek-ai/Janus) for releasing pretrained checkpoints for evaluation.

## 📄 Citation

```
@article{jiao2025unitoken,
  title={UniToken: Harmonizing Multimodal Understanding and Generation through Unified Visual Encoding},
  author={Jiao, Yang and Qiu, Haibo and Jie, Zequn and Chen, Shaoxiang and Chen, Jingjing and Ma, Lin and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2504.04423},
  year={2025}
}
```
