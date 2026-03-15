# controlnet_interior_v4_bedroom.py
# Interior redesign — bedroom focused, better structure preservation, anti-text, anti-wrong-furniture

import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from transformers import (
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
    CLIPProcessor,
    CLIPModel,
)
import cv2


# ─── ADE20K color palette ─────────────────────────────────────────────────────

def ade_palette():
    return [
        [120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],
        [120,120,80],[140,140,140],[204,5,255],[230,230,230],[4,250,7],
        [224,5,255],[235,255,7],[150,5,61],[120,120,70],[8,255,51],
        [255,6,82],[143,255,140],[204,255,4],[255,51,7],[204,70,3],
        [0,102,200],[61,230,250],[255,6,51],[11,102,255],[255,7,71],
        [255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],
        [8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],
        [7,255,255],[224,255,8],[102,8,255],[255,61,6],[255,194,7],
    ]


# ─── Bedroom style presets ────────────────────────────────────────────────────

BEDROOM_STYLES = {
    "modern": {
        "prompt": (
            "modern minimalist bedroom, low platform bed with white linen, "
            "two matching nightstands, clean architectural lines, "
            "white and light grey walls, light oak hardwood floor, "
            "recessed LED lighting, large window with natural daylight, "
            "no extra furniture, no tables, no chairs, no clutter, "
            "professional interior photography, 8k, ultra realistic, sharp focus"
        ),
        "negative_prompt": (
            "dining table, coffee table, office table, chairs, dining chairs, "
            "wrong furniture, misplaced furniture, kitchen, bathroom, living room, "
            "text, writing, letters, words, numbers, watermark, signature, logo, "
            "typography, font, alphabet, characters, symbols, inscriptions, "
            "person, people, face, clothes, dress, laundry, clutter, extra objects, "
            "deformed, blurry, low quality, cartoon, 3d render, painting, "
            "oversaturated, dark, ugly, busy patterns"
        ),
    },
    "scandinavian": {
        "prompt": (
            "scandinavian bedroom, simple wooden bed frame with white bedding, "
            "minimal nightstand, warm white walls, light pine wood floor, "
            "soft natural lighting, cozy hygge atmosphere, simple textiles, "
            "no extra furniture, no clutter, "
            "professional interior photography, 8k, ultra realistic, sharp focus"
        ),
        "negative_prompt": (
            "dining table, coffee table, office table, chairs, dining chairs, "
            "wrong furniture, misplaced furniture, kitchen, bathroom, living room, "
            "text, writing, letters, words, numbers, watermark, signature, logo, "
            "typography, font, alphabet, characters, symbols, inscriptions, "
            "person, people, face, clothes, dress, laundry, clutter, extra objects, "
            "deformed, blurry, low quality, cartoon, 3d render, painting, "
            "dark, cold, industrial, busy patterns"
        ),
    },
    "industrial": {
        "prompt": (
            "industrial bedroom, metal bed frame with dark bedding, "
            "exposed brick wall, concrete floor, Edison bulb lighting, "
            "dark metal accents, raw materials, urban loft feel, "
            "no extra furniture, no clutter, "
            "professional interior photography, 8k, ultra realistic, sharp focus"
        ),
        "negative_prompt": (
            "dining table, coffee table, office table, chairs, dining chairs, "
            "wrong furniture, misplaced furniture, kitchen, bathroom, living room, "
            "text, writing, letters, words, numbers, watermark, signature, logo, "
            "typography, font, alphabet, characters, symbols, inscriptions, "
            "person, people, face, clothes, dress, laundry, clutter, extra objects, "
            "deformed, blurry, low quality, cartoon, 3d render, painting, "
            "bright, pastel, floral, feminine"
        ),
    },
    "bohemian": {
        "prompt": (
            "bohemian bedroom, low wooden bed with layered colorful textiles, "
            "warm earthy tones, terracotta walls, woven rugs, macrame wall hanging, "
            "warm ambient lighting, plants, eclectic decor, cozy and relaxed atmosphere, "
            "no extra furniture, no clutter, "
            "professional interior photography, 8k, ultra realistic, sharp focus"
        ),
        "negative_prompt": (
            "dining table, coffee table, office table, chairs, dining chairs, "
            "wrong furniture, misplaced furniture, kitchen, bathroom, living room, "
            "text, writing, letters, words, numbers, watermark, signature, logo, "
            "typography, font, alphabet, characters, symbols, inscriptions, "
            "person, people, face, clothes, dress, laundry, clutter, extra objects, "
            "deformed, blurry, low quality, cartoon, 3d render, painting, "
            "cold, industrial, minimalist, white walls"
        ),
        
},
    "declutter": {
    "prompt": (
        "clean tidy bedroom, same room same furniture same layout, "
        "no laundry, no clothes, no clutter, everything in its place, "
        "beds made with clean linen, floor clear, "
        "same lighting same walls same furniture same colors, "
        "photorealistic, 8k, sharp focus"
    ),
    "negative_prompt": (
        "laundry, clothes, shirts, pants, socks, underwear, fabric on floor, "
        "fabric on chair, messy bed, unmade bed, clutter, scattered objects, "
        "text, watermark, person, people, face, "
        "style change, different furniture, different walls, different colors, "
        "deformed, blurry, low quality, cartoon"
    ),
    "clip_prompt": "clean tidy bedroom, no clutter, no laundry, same room photorealistic",

    },
}


# ─── Load models ─────────────────────────────────────────────────────────────

def load_models(device="cuda"):
    print("Loading segmentation model...")
    seg_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    seg_model = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small"
    ).to(device)

    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "BertChristiaens/controlnet-seg-room",
        torch_dtype=torch.float16,
    )

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled")
    except Exception:
        print("xformers not available, continuing without it")

    print("Loading CLIP scorer...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return seg_processor, seg_model, pipe, clip_model, clip_processor


# ─── Segmentation ─────────────────────────────────────────────────────────────

def segment_room(image, seg_processor, seg_model, device="cuda"):
    pixel_values = seg_processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = seg_model(pixel_values)

    seg_map = seg_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0].cpu().numpy()

    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    palette = ade_palette()
    for label, color in enumerate(palette):
        color_seg[seg_map == label] = color

    # Morphological smoothing — cleaner region boundaries for ControlNet
    kernel = np.ones((5, 5), np.uint8)
    color_seg = cv2.morphologyEx(color_seg, cv2.MORPH_CLOSE, kernel)
    color_seg = cv2.morphologyEx(color_seg, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(color_seg)


# ─── CLIP scoring ─────────────────────────────────────────────────────────────

def score_image(image, prompt, clip_model, clip_processor, device="cuda"):
    inputs = clip_processor(
        text=[prompt], images=image, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        score = clip_model(**inputs).logits_per_image.item()
    return score


# ─── Main pipeline ────────────────────────────────────────────────────────────

def redesign_bedroom(
    image_path,
    style="industrial",
    extra_prompt="",
    num_inference_steps=60,
    guidance_scale=8.5,
    controlnet_conditioning_scale=1.4,
    num_candidates=1,
    second_pass=False,
    device="cuda",
):
    seg_processor, seg_model, pipe, clip_model, clip_processor = load_models(device)

    image = load_image(image_path).convert("RGB")
    image = image.resize((768, 768), Image.LANCZOS)

    style_config = BEDROOM_STYLES[style]
    prompt = style_config["prompt"]
    if extra_prompt:
        prompt = prompt + ", " + extra_prompt
    negative_prompt = style_config["negative_prompt"]

    print("Segmenting room...")
    seg_image = segment_room(image, seg_processor, seg_model, device)

    print(f"Generating {num_candidates} candidates...")
    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=seg_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_images_per_prompt=num_candidates,
        width=768,
        height=768,
    ).images

    print("Scoring with CLIP...")
    scores = [score_image(img, prompt, clip_model, clip_processor, device) for img in results]
    best = results[scores.index(max(scores))]

    if second_pass:
        print("Refining best result...")
        refine_pipe = StableDiffusionImg2ImgPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(device)

        best = refine_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=best,
            strength=0.25,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
        ).images[0]

    best.save("output_best.png")
    seg_image.save("segmentation_map.png")

    comparison = Image.new("RGB", (image.width * 2 + 20, image.height), (255, 255, 255))
    comparison.paste(image, (0, 0))
    comparison.paste(best.resize((image.width, image.height)), (image.width + 20, 0))
    comparison.save("comparison.png")

    print(f"Done. Best CLIP score: {max(scores):.2f}")
    return best


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = redesign_bedroom(
        image_path="/content/WhatsApp-Image-2024-08-24-at-2.35.48-PM-scaled-qt8n367jlzs0pncar898ktn04kwnogwql93let8y28.jpeg.webp",
        style="industrial",
        extra_prompt="with a reading nook",
        second_pass=True,
    )
