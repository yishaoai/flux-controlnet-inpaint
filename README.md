[200~### 介绍

本项目主要是介绍如何结合 flux 和 controlnet 进行 inpaint，以童装场景为示例。更多详细的介绍请参考：[yishaoai/tutorials-of-100-wonderful-ai-models](https://github.com/yishaoai/tutorials-of-100-wonderful-ai-models/)的第三节。

本项目的使用方法如下：
```python
import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from controlnet_aux import CannyDetector

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'YishaoAI/flux-dev-controlnet-canny-kid-clothes'

pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.to("cuda")

image = load_image(image_path)
mask = load_image(mask_path)
canny = CannyDetector()
canny_image = canny(image)

image_res = pipe(
    prompt,
    image=image,
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,
    mask_image=mask,
    strength=0.95,
    num_inference_steps=50,
    guidance_scale=5,
    generator=generator,
    joint_attention_kwargs={"scale": 0.8},
    ).images[0]
```

### 结果示例
以下示例图是以童装场景为例，会把衣服部分保持不变，将人像和背景基于提示词进行重绘。因为用到controlnet，所以这些图像的边缘会相似。基于需要controlnet的权重可以进行调整，权重越大，生成图像的边缘信息保留越多，权重越小，边缘信息保留越少。
inpaint方法适合对爆款产品进行复刻，十分适合电商领域的买家秀和种草的用户需求。同时，也可以给商家朋友提供服装模特的主图。

![demo](https://github.com/yishaoai/flux-controlnet-inpaint/assets/flux-controlnet-inpaint.png)

