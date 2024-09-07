# Introduction

[中文版本](https://github.com/yishaoai/flux-controlnet-inpaint/blob/main/README_ZH.md)
This project mainly introduces how to combine flux and controlnet for inpaint, taking the children's clothing scene as an example. For more detailed introduction, please refer to the third section of [yishaoai/tutorials-of-100-wonderful-ai-models](https://github.com/yishaoai/tutorials-of-100-wonderful-ai-models/).

The usage of this project is as follows:

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
prompt = "children's clothing model"

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
### Result example
The following example image is based on the children's clothing scene. The clothing part will remain unchanged, and the portrait and background will be redrawn based on the prompt word. Because controlnet is used, the edges of these images will be similar. The weight of controlnet can be adjusted based on the need. The larger the weight, the more edge information of the generated image is retained, and the smaller the weight, the less edge information is retained.
The inpaint method is suitable for reproducing popular products, which is very suitable for the needs of buyers' shows and users in the e-commerce field. At the same time, it can also provide merchants with the main picture of clothing models.


![demo](https://github.com/yishaoai/flux-controlnet-inpaint/blob/main/assets/flux-controlnet-inpaint.png)

