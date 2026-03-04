# controlnet_interior_v3.py
# Interior redesign using ControlNet + Realistic Vision V5.1 + CLIP auto-picker

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

def ade_palette():
    return [[120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],
            [120,120,80],[140,140,140],[204,5,255],[230,230,230],[4,250,7],
            [224,5,255],[235,255,7],[150,5,61],[120,120,70],[8,255,51],
            [255,6,82],[143,255,140],[204,255,4],[255,51,7],[204,70,3],
            [0,102,200],[61,230,250],[255,6,51],[11,102,255],[255,7,71],
            [255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],
            [8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],
            [7,255,255],[224,255,8],[102,8,255],[255,61,6],[255,194,7]]

# Load segmentation model
seg_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
seg_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

# Load ControlNet + Realistic Vision V5.1
controlnet = ControlNetModel.from_pretrained(
    "BertChristiaens/controlnet-seg-room",
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

# Load CLIP scorer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load your image
image = load_image("/content/360_F_256201353_cl925V1nGNPwkNAEq0APgAevUymgaZh3-2.jpg").resize((512, 512))

# Segmentation
pixel_values = seg_processor(image, return_tensors="pt").pixel_values
with torch.no_grad():
    outputs = seg_model(pixel_values)
seg_map = seg_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0].numpy()
color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(ade_palette()):
    color_seg[seg_map == label] = color
seg_image = Image.fromarray(color_seg)

# Prompts
prompt = "modern bedroom, platform bed, clean lines, grey and white walls, hardwood floor, minimal furniture, recessed lighting, natural light, interior design photography, 4k, photorealistic"
negative_prompt = "text, writing, letters, words, numbers, watermark, person, people, face, clothes, dress, laundry, clutter, extra objects, deformed, blurry, low quality, cartoon, 3d render"

# Generate 4 options
results = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=seg_image,
    num_inference_steps=50,
    guidance_scale=6.0,
    controlnet_conditioning_scale=1.5,
    num_images_per_prompt=4,
).images

# CLIP picks the best one automatically
clip_prompt = "modern bedroom interior design, photorealistic, high quality"
scores = []
for img in results:
    inputs = clip_processor(text=[clip_prompt], images=img, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        score = clip_model(**inputs).logits_per_image.item()
    scores.append(score)

best = results[scores.index(max(scores))]
best.save("output_best.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image); axes[0].set_title("Before")
axes[1].imshow(best); axes[1].set_title(f"Best (score: {max(scores):.1f})")
plt.savefig("comparison.png")
plt.show()
