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


def base64_to_image(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data)).convert("RGBA")


def bytes_to_base64(data: bytes, fmt="PNG") -> str:
    return base64.b64encode(data).decode("utf-8")


async def download_image(url: str) -> Image.Image:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")


def select_best_mask(masks: list, image_width: int, image_height: int) -> np.ndarray:
    """
    Selects the most prominent mask: largest area, preferring objects near the center.
    """
    cx, cy = image_width / 2, image_height / 2
    best_score = -1
    best_mask = None

    for mask_data in masks:
        arr = np.array(mask_data)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        area = np.sum(arr > 128)
        if area == 0:
            continue
        ys, xs = np.where(arr > 128)
        mask_cx = np.mean(xs)
        mask_cy = np.mean(ys)
        dist = np.sqrt((mask_cx - cx) ** 2 + (mask_cy - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        centrality = 1 - (dist / max_dist)
        # Balance area and centrality
        total_pixels = image_width * image_height
        area_score = area / total_pixels
        score = 0.6 * area_score + 0.4 * centrality
        if score > best_score:
            best_score = score
            best_mask = arr

    return best_mask


def dilate_mask(mask: np.ndarray, pixels: int = 20) -> Image.Image:
    """Dilates the mask so the fabric wrapping slightly overflows the object edges."""
    mask_img = Image.fromarray((mask > 128).astype(np.uint8) * 255, mode="L")
    for _ in range(pixels // 5):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(11))
    return mask_img


def build_inpaint_prompt(material: str) -> tuple[str, str]:
    prompts = {
        "tela": (
            "The object is completely wrapped and covered in flowing white linen fabric, "
            "tied with thick rope, photorealistic, Christo and Jeanne-Claude art installation, "
            "soft folds and wrinkles in cloth, dramatic natural lighting, 8K detail, "
            "fabric texture visible, seamless wrapping, no exposed parts of original object",
            "ugly, blurry, low quality, deformed, text, watermark, cartoon, painting, exposed object"
        ),
        "plastico": (
            "The object is completely wrapped in shiny transparent plastic wrap and polypropylene, "
            "tied with nylon rope, photorealistic, Christo and Jeanne-Claude art installation, "
            "reflections on plastic surface, dramatic lighting, 8K detail",
            "ugly, blurry, low quality, deformed, text, watermark, cartoon"
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

    # Resize if too large (keep aspect ratio, max 1024px)
    max_size = 1024
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        original_image = original_image.resize((new_w, new_h), Image.LANCZOS)
        width, height = new_w, new_h

    # Encode image as base64 URI for Replicate
    img_b64 = image_to_base64(original_image)
    img_data_uri = f"data:image/png;base64,{img_b64}"

    # 2. Run SAM automatic segmentation
    try:
        sam_output = replicate.run(
            "schannel/segment-anything-2:d231fe4dc7c168e2dc72fb1ded4f647aba62f6e07e44e01b5b1b3fecc5a21bfe",
            input={
                "image": img_data_uri,
                "use_m2m": True,
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en segmentacion SAM: {str(e)}")

    # 3. Download and select best mask
    masks = []
    if isinstance(sam_output, list):
        for item in sam_output:
            url = item if isinstance(item, str) else getattr(item, "url", None)
            if url:
                mask_img = await download_image(url)
                masks.append(np.array(mask_img.convert("L")))

    if not masks:
        raise HTTPException(status_code=500, detail="No se detectaron objetos en la imagen")

    best_mask_arr = select_best_mask(masks, width, height)
    if best_mask_arr is None:
        raise HTTPException(status_code=500, detail="No se pudo seleccionar un objeto principal")

    # 4. Dilate mask for better wrapping coverage
    mask_pil = dilate_mask(best_mask_arr, pixels=25)
    mask_b64 = image_to_base64(mask_pil, fmt="PNG")
    mask_data_uri = f"data:image/png;base64,{mask_b64}"

    # 5. Run Stable Diffusion XL Inpainting
    prompt, negative_prompt = build_inpaint_prompt(material)

    try:
        inpaint_output = replicate.run(
            "stability-ai/stable-diffusion-xl-base-1.0:a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c",
            input={
                "image": img_data_uri,
                "mask": mask_data_uri,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": 50,
                "guidance_scale": 8.5,
                "strength": 0.99,
                "scheduler": "DPMSolverMultistep",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en generacion de imagen: {str(e)}")

    # 6. Get result image
    result_url = None
    if isinstance(inpaint_output, list) and len(inpaint_output) > 0:
        result_url = inpaint_output[0]
        if hasattr(result_url, "url"):
            result_url = result_url.url
    elif isinstance(inpaint_output, str):
        result_url = inpaint_output

    if not result_url:
        raise HTTPException(status_code=500, detail="No se obtuvo imagen de resultado")

    result_image = await download_image(result_url)
    result_b64 = image_to_base64(result_image)

    # Also return the mask for debug purposes
    mask_result_b64 = image_to_base64(mask_pil)

    return JSONResponse({
        "result_image": result_b64,
        "mask_image": mask_result_b64,
        "width": result_image.width,
        "height": result_image.height,
    })
