import os
from typing import Optional
import torch
import numpy as np
import cv2
import tempfile
from PIL import Image
from cog import BasePredictor, Input, Path, Secret
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from transformers import pipeline as hf_pipeline


STYLES = {
    "bedroom": {
        "modern": {
            "prompt": (
                "A contemporary modern bedroom. Upholstered bed with curved headboard in warm grey velvet, "
                "sculptural bedside lamps, textured wall paneling, deep walnut wood floor, "
                "layered neutral textiles, ambient warm lighting, statement artwork above bed. "
                "Rich materials, considered design, no clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, dining table, coffee table, chairs, "
                "wrong furniture, kitchen, bathroom, person, people, face, clothes, laundry, clutter, "
                "blurry, low quality, cartoon, cold, sterile, plain white walls"
            ),
        },
        "minimalist": {
            "prompt": (
                "A pure minimalist bedroom. Ultra low platform bed with white and off-white linen, "
                "simple white nightstand, bare white walls, light bleached oak or pale concrete floor, "
                "single recessed light, no decorations, nothing on surfaces, "
                "all white and off-white palette, only small black accents if any. "
                "Architectural photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, dining table, chairs, wrong furniture, "
                "kitchen, bathroom, person, people, face, clothes, laundry, clutter, decorations, art, plants, "
                "blurry, low quality, cartoon, busy, colorful, dark colors, black sofa, black bed, "
                "brown, grey, beige, warm tones, headboard"
            ),
        },
        "scandinavian": {
            "prompt": (
                "A scandinavian hygge bedroom. Simple birch bed frame with white bedding, "
                "warm white walls, pine wood floor, soft pendant lamp, small plant. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, dining table, chairs, wrong furniture, "
                "kitchen, bathroom, person, people, face, clothes, laundry, clutter, "
                "blurry, low quality, cartoon, cold, industrial"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial loft bedroom. Black metal bed frame, exposed brick wall, "
                "polished concrete floor, Edison pendant bulbs, black iron shelving. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, dining table, chairs, wrong furniture, "
                "kitchen, bathroom, person, people, face, clothes, laundry, clutter, "
                "blurry, low quality, cartoon, pastel, floral"
            ),
        },
        "bohemian": {
            "prompt": (
                "A bohemian bedroom. Low wooden bed with layered terracotta textiles, "
                "woven rug, macrame wall hanging, brass pendant, trailing plants, earthy walls. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, dining table, chairs, wrong furniture, "
                "kitchen, bathroom, person, people, face, clothes, laundry, clutter, "
                "blurry, low quality, cartoon, cold, sterile"
            ),
        },
    },
    "living_room": {
        "modern": {
            "prompt": (
                "A contemporary modern living room. Large curved sectional sofa in warm bouclé, "
                "sculptural marble coffee table, warm oak floor, textured plaster walls, "
                "dramatic pendant light, layered rugs, curated art and objects. "
                "Rich and considered design, no clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, bathroom, kitchen, wrong furniture, "
                "person, people, face, clutter, blurry, low quality, cartoon, plain white walls, sterile"
            ),
        },
        "minimalist": {
            "prompt": (
                "A pure minimalist living room. Single low sofa in white or off-white linen, "
                "one small white side table, bare white walls, pale concrete or light oak floor, "
                "no art, no plants, no decorations, nothing on surfaces, "
                "one thin white floor lamp, complete emptiness. "
                "All white and off-white palette, only small black accents if any. "
                "Architectural photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, bathroom, kitchen, wrong furniture, "
                "person, people, face, clutter, decorations, art, plants, rugs, "
                "blurry, low quality, cartoon, busy, colorful, dark colors, black sofa, "
                "brown, grey, beige, warm tones"
            ),
        },
        "scandinavian": {
            "prompt": (
                "A scandinavian living room. Linen sofa, birch coffee table, white walls, "
                "oak floor, woolen throw, small plant. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, bathroom, kitchen, wrong furniture, "
                "person, people, face, clutter, blurry, low quality, cartoon, industrial"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial living room. Dark leather sofa, steel coffee table, "
                "concrete wall, Edison bulbs, metal shelving. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, bathroom, kitchen, wrong furniture, "
                "person, people, face, clutter, blurry, low quality, cartoon, pastel"
            ),
        },
        "bohemian": {
            "prompt": (
                "A bohemian living room. Colorful cushions, kilim rugs, macrame, rattan, "
                "trailing plants, string lights, terracotta walls. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, bathroom, kitchen, wrong furniture, "
                "person, people, face, clutter, blurry, low quality, cartoon, sterile"
            ),
        },
    },
    "bathroom": {
        "modern": {
            "prompt": (
                "A contemporary modern bathroom. Freestanding sculptural soaking tub, "
                "large format warm stone tiles, floating double vanity, brushed gold fixtures, "
                "backlit frameless mirror, warm ambient lighting, single large plant. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, kitchen, "
                "person, people, face, toiletries mess, clutter, "
                "blurry, low quality, cartoon, cold, sterile, white tiles"
            ),
        },
        "minimalist": {
            "prompt": (
                "A pure minimalist bathroom. Simple white rectangular basin on a white shelf, "
                "white or off-white walls, pale concrete or white tile floor, single thin mirror, "
                "one recessed light, no decorations, no plants, no toiletries visible. "
                "All white and off-white palette, thin black fixtures as only accent. "
                "Architectural photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, kitchen, "
                "person, people, face, clutter, decorations, plants, toiletries, "
                "blurry, low quality, cartoon, colorful, dark colors, black surfaces, "
                "brown, beige, warm tones, busy"
            ),
        },
        "spa": {
            "prompt": (
                "A luxury spa bathroom. Travertine stone, deep soaking tub, rainfall shower, "
                "teak accents, ambient lighting, tropical plants. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, kitchen, "
                "person, people, face, clutter, blurry, low quality, cartoon, industrial"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial bathroom. Concrete walls, black fixtures, vessel sink, "
                "Edison mirror, black hexagon tiles. No clutter. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, kitchen, "
                "person, people, face, clutter, blurry, low quality, cartoon, pastel, floral"
            ),
        },
    },
    "kitchen": {
        "modern": {
            "prompt": (
                "A contemporary modern kitchen. Handleless cabinets in deep charcoal and warm oak, "
                "thick marble island with waterfall edge, statement pendant lights, "
                "integrated appliances, warm under-cabinet lighting, herringbone tile backsplash. "
                "Counters clear, considered design. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, bathroom, "
                "person, people, face, dirty dishes, clutter, "
                "blurry, low quality, cartoon, rustic, all white, plain"
            ),
        },
        "minimalist": {
            "prompt": (
                "A pure minimalist kitchen. Seamless white handleless cabinets floor to ceiling, "
                "thin white quartz countertop, no island, hidden integrated appliances, "
                "single strip of recessed light, no objects on any surface, "
                "white walls, pale concrete or white tile floor. "
                "All white and off-white palette, thin black handles as only accent. "
                "Architectural photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, bathroom, "
                "person, people, face, dirty dishes, clutter, decorations, plants, "
                "blurry, low quality, cartoon, rustic, colorful, dark colors, black cabinets, "
                "brown, beige, warm tones, busy"
            ),
        },
        "rustic": {
            "prompt": (
                "A rustic farmhouse kitchen. Shaker oak cabinets, butcher block counters, "
                "farmhouse sink, open shelving, exposed beams. Counters clear. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, bathroom, "
                "person, people, face, dirty dishes, clutter, "
                "blurry, low quality, cartoon, modern, cold"
            ),
        },
        "industrial": {
            "prompt": (
                "An industrial kitchen. Stainless steel counters, black cabinets, "
                "exposed brick, metal shelving, Edison bulbs. Counters clear. "
                "Interior photography, sharp focus, 8k, photorealistic."
            ),
            "negative_prompt": (
                "text, writing, letters, words, numbers, watermark, signature, logo, typography, "
                "font, alphabet, symbols, inscriptions, bed, bedroom, living room, bathroom, "
                "person, people, face, dirty dishes, clutter, "
                "blurry, low quality, cartoon, pastel"
            ),
        },
    },
}


class Predictor(BasePredictor):



    def setup(self):
        self.depth_estimator = None
        self.pipe = None
        self._models_loaded = False
        self._load_models()

    def _load_models(self, hf_token: Optional[str] = None):
        if self._models_loaded:
            return

        # Accept token from Secret input first, then environment fallback.
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_APIKEY")

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        print("Loading depth estimator...")
        self.depth_estimator = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0,
        )

        print("Loading ControlNet Union Pro 2.0...")
        controlnet_union = FluxControlNetModel.from_pretrained(
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        controlnet = FluxMultiControlNetModel([controlnet_union])

        print("Loading FLUX schnell pipeline...")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()

        self._models_loaded = True
        print("Setup complete.")

    def predict(
        self,
        image: Path = Input(description="Photo of your room"),
        room_type: str = Input(
            description="Type of room",
            choices=["bedroom", "living_room", "bathroom", "kitchen"],
            default="bedroom",
        ),
        style: str = Input(
            description="Design style. bedroom/living_room: modern|minimalist|scandinavian|industrial|bohemian. bathroom: modern|minimalist|spa|industrial. kitchen: modern|minimalist|rustic|industrial",
            choices=["modern", "minimalist", "scandinavian", "industrial", "bohemian", "spa", "rustic"],
            default="minimalist",
        ),
        extra_prompt: str = Input(
            description="Optional extra details e.g. 'with a fireplace' or 'warm lighting'",
            default="",
        ),
        hf_token: Optional[Secret] = Input(
            description="Optional Hugging Face token (hf_...) for gated model access",
            default=None,
        ),
    ) -> Path:

        token_value = hf_token.get_secret_value() if hf_token else None
        try:
            self._load_models(token_value)
        except Exception as e:
            raise RuntimeError(
                "Model initialization failed. If access is gated, provide hf_token as a Secret input "
                "or set HF_TOKEN in environment."
            ) from e

        if style not in STYLES[room_type]:
            available = list(STYLES[room_type].keys())
            raise ValueError(f"Style '{style}' not available for '{room_type}'. Choose from: {available}")

        input_image = Image.open(str(image)).convert("RGB").resize((640, 640), Image.LANCZOS)

        style_config    = STYLES[room_type][style]
        prompt          = style_config["prompt"]
        if extra_prompt.strip():
            prompt = prompt + " " + extra_prompt.strip() + "."
        negative_prompt = style_config["negative_prompt"]

        print(f"Room: {room_type} | Style: {style}")

        # Depth map
        depth = self.depth_estimator(input_image)["depth"]
        depth_image = depth.convert("RGB").resize(input_image.size)

        # Canny edges
        img_array = np.array(input_image)
        gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges     = cv2.Canny(gray, 80, 180)
        canny_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        # Generate
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=[depth_image, canny_image],
            control_mode=[2, 0],
            controlnet_conditioning_scale=[0.6, 0.4],
            control_guidance_end=0.6,
            num_inference_steps=4,
            guidance_scale=0,
            height=640,
            width=640,
        ).images[0]

        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        result.save(tmp)
        return Path(tmp)