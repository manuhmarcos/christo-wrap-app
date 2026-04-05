import os
import base64
import httpx
import replicate
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Christo Wrapping API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


def image_to_base64(img: Image.Image, fmt="PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


async def download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def download_image(url: str) -> Image.Image:
    data = await download_bytes(url)
    return Image.open(BytesIO(data)).convert("RGB")


def create_center_mask(width: int, height: int, coverage: float = 0.6) -> Image.Image:
    """Creates an elliptical mask covering the center of the image."""
    mask = Image.new("L", (width, height), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    margin_x = int(width * (1 - coverage) / 2)
    margin_y = int(height * (1 - coverage) / 2)
    draw.ellipse([margin_x, margin_y, width - margin_x, height - margin_y], fill=255)
    # Blur edges for smoother transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=30))
    return mask


def dilate_mask(mask_arr: np.ndarray, pixels: int = 20) -> Image.Image:
    mask_img = Image.fromarray((mask_arr > 128).astype(np.uint8) * 255, mode="L")
    for _ in range(pixels // 5):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(11))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=5))
    return mask_img


def build_inpaint_prompt(material: str) -> tuple[str, str]:
    prompts = {
        "tela": (
            "object completely wrapped and covered in flowing white fabric cloth, "
            "tied with thick rope and twine, photorealistic, Christo and Jeanne-Claude "
            "art installation style, soft natural fabric folds and wrinkles, "
            "dramatic natural lighting, ultra detailed 8K photography, "
            "seamless wrapping covering entire object",
            "ugly, blurry, low quality, deformed, text, watermark, cartoon, "
            "painting, illustration, fake, unrealistic, exposed object underneath"
        ),
        "plastico": (
            "object completely wrapped in shiny polypropylene plastic sheeting, "
            "tied with thick nylon rope, photorealistic, Christo and Jeanne-Claude "
            "art installation style, plastic reflections and highlights, "
            "dramatic lighting, ultra detailed 8K photography",
            "ugly, blurry, low quality, deformed, text, watermark, cartoon, painting"
        ),
    }
    return prompts.get(material, prompts["tela"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/wrap")
async def wrap_object(
    file: UploadFile = File(...),
    material: str = "tela"
):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN no configurado")

    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

    # 1. Read and validate image
    contents = await file.read()
    try:
        original_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen invalida")

    width, height = original_image.size

    # Resize to 1024x1024 max (keep aspect ratio)
    max_size = 1024
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        original_image = original_image.resize((new_w, new_h), Image.LANCZOS)
        width, height = new_w, new_h

    # Make dimensions multiples of 8 (required by SD)
    width = (width // 8) * 8
    height = (height // 8) * 8
    original_image = original_image.resize((width, height), Image.LANCZOS)

    img_b64 = image_to_base64(original_image)
    img_data_uri = f"data:image/png;base64,{img_b64}"

    # 2. Run SAM 2 for automatic segmentation
    mask_arr = None
    try:
        sam_output = replicate.run(
            "meta/sam-2",
            input={
                "image": img_data_uri,
                "use_m2m": True,
                "points_per_side": 32,
                "pred_iou_thresh": 0.86,
                "stability_score_thresh": 0.92,
            }
        )

        # Parse SAM output - returns list of mask URLs
        masks = []
        output_list = list(sam_output) if not isinstance(sam_output, list) else sam_output

        for item in output_list:
            url = item if isinstance(item, str) else getattr(item, "url", str(item))
            if url and url.startswith("http"):
                try:
                    mask_img = await download_image(url)
                    arr = np.array(mask_img.convert("L"))
                    area = np.sum(arr > 128)
                    if area > 0:
                        masks.append(arr)
                except Exception:
                    continue

        if masks:
            # Select largest mask near center
            cx, cy = width / 2, height / 2
            best_score = -1
            for arr in masks:
                area = np.sum(arr > 128)
                ys, xs = np.where(arr > 128)
                mcx, mcy = np.mean(xs), np.mean(ys)
                dist = np.sqrt((mcx - cx) ** 2 + (mcy - cy) ** 2)
                max_dist = np.sqrt(cx**2 + cy**2)
                centrality = 1 - (dist / max_dist)
                score = 0.6 * (area / (width * height)) + 0.4 * centrality
                if score > best_score:
                    best_score = score
                    mask_arr = arr

    except Exception as e:
        # SAM failed, fall back to center ellipse mask
        mask_arr = None

    # 3. Build mask (from SAM or fallback ellipse)
    if mask_arr is not None:
        mask_pil = dilate_mask(mask_arr, pixels=30)
    else:
        # Fallback: use center ellipse mask
        mask_pil = create_center_mask(width, height, coverage=0.65)

    mask_b64 = image_to_base64(mask_pil, fmt="PNG")
    mask_data_uri = f"data:image/png;base64,{mask_b64}"

    # 4. Run SDXL Inpainting
    prompt, negative_prompt = build_inpaint_prompt(material)

    try:
        inpaint_output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "image": img_data_uri,
                "mask": mask_data_uri,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": 50,
                "guidance_scale": 8.5,
                "strength": 0.99,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en generacion de imagen: {str(e)}")

    # 5. Get result image
    result_url = None
    output_list = list(inpaint_output) if not isinstance(inpaint_output, list) else inpaint_output
    if output_list:
        first = output_list[0]
        result_url = first if isinstance(first, str) else getattr(first, "url", str(first))

    if not result_url or not result_url.startswith("http"):
        raise HTTPException(status_code=500, detail="No se obtuvo imagen de resultado")

    result_bytes = await download_bytes(result_url)
    result_image = Image.open(BytesIO(result_bytes)).convert("RGB")
    result_b64 = image_to_base64(result_image)
    mask_result_b64 = image_to_base64(mask_pil)

    return JSONResponse({
        "result_image": result_b64,
        "mask_image": mask_result_b64,
        "width": result_image.width,
        "height": result_image.height,
    })
