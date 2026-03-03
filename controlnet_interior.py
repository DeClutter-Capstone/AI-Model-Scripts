# ===== CELL 1 — Install =====
!pip install -q diffusers transformers accelerate xformers opencv-python

# ===== CELL 2 — Imports =====
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# ===== CELL 3 — Load segmentation model (extracts room structure) =====
seg_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
seg_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

# ===== CELL 4 — Load ControlNet + Stable Diffusion =====
controlnet = ControlNetModel.from_pretrained(
    "BertChristiaens/controlnet-seg-room",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

# ===== CELL 5 — Load your image =====
image = load_image("/content/WhatsApp-Image-2024-08-24-at-2.35.48-PM-scaled-qt8n367jlzs0pncar898ktn04kwnogwql93let8y28.jpeg.webp").resize((512, 512))

# CELL 6 FIXED — proper color segmentation map

import numpy as np
from PIL import Image
import torch

# ADE20K palette (what this ControlNet was trained on)
def ade_palette():
    return [[120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],
            [120,120,80],[140,140,140],[204,5,255],[230,230,230],[4,250,7],
            [224,5,255],[235,255,7],[150,5,61],[120,120,70],[8,255,51],
            [255,6,82],[143,255,140],[204,255,4],[255,51,7],[204,70,3],
            [0,102,200],[61,230,250],[255,6,51],[11,102,255],[255,7,71],
            [255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],
            [8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],
            [7,255,255],[224,255,8],[102,8,255],[255,61,6],[255,194,7]]

pixel_values = seg_processor(image, return_tensors="pt").pixel_values
with torch.no_grad():
    outputs = seg_model(pixel_values)

seg_map = seg_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0].numpy()

# Convert label map to colors
palette = ade_palette()
color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(palette):
    color_seg[seg_map == label] = color

seg_image = Image.fromarray(color_seg)
seg_image.save("seg_map.png")
seg_image  # show it — should look like a colored room outline

# ===== CELL 7 — Generate redesigned room =====
prompt = "modern bedroom, platform bed, clean lines, grey and white walls, hardwood floor, minimal furniture, recessed lighting, large window, natural light, interior design photography, 4k, photorealistic"

negative_prompt = "text, writing, letters, words, numbers, watermark, signature, logo, person, people, face, body, clothes, dress, laundry, clutter, mess, extra objects, floating objects, deformed, distorted, blurry, low quality, grainy, cartoon, painting, 3d render, colorful, ornate, busy, vintage, rustic"

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=seg_image,
    num_inference_steps=50,
    guidance_scale=6.0,
    controlnet_conditioning_scale=1.2,
).images[0]

result.save("output_modern.png")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image); axes[0].set_title("Before")
axes[1].imshow(result); axes[1].set_title("After — Modern Bedroom")
plt.savefig("comparison.png")
plt.show()
