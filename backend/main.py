import os
import base64
import httpx
import replicate
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import cv2
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
SD_SIZE = 512


def image_to_base64(img: Image.Image, fmt="PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_data_uri(img: Image.Image, fmt="PNG") -> str:
    return f"data:image/png;base64,{image_to_base64(img, fmt)}"


async def download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def segment_object_grabcut(img_pil: Image.Image) -> Image.Image:
    """
    Uses OpenCV GrabCut to segment the main object (no API call needed).
    Returns a binary mask as PIL Image (L mode, 255=object, 0=background).
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # Define rect around center (object is assumed to be roughly centered)
    margin_x = int(w * 0.12)
    margin_y = int(h * 0.08)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
        # Pixels marked as foreground or probable foreground
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        # Fallback: center ellipse
        fg_mask = np.zeros((h, w), np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.35), int(h * 0.42))
        cv2.ellipse(fg_mask, center, axes, 0, 0, 360, 255, -1)

    # Morphological cleanup: close small holes, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Slight dilation so fabric slightly overflows edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)

    # Smooth edges with gaussian blur for natural blending
    fg_mask_blur = cv2.GaussianBlur(fg_mask, (21, 21), 0)

    return Image.fromarray(fg_mask_blur, mode="L")


def resize_for_sd(img: Image.Image):
    """Resize keeping aspect ratio, pad to SD_SIZE x SD_SIZE."""
    w, h = img.size
    ratio = SD_SIZE / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (SD_SIZE, SD_SIZE), (0, 0, 0))
    ox = (SD_SIZE - new_w) // 2
    oy = (SD_SIZE - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, (ox, oy, new_w, new_h)


def resize_mask_for_sd(mask: Image.Image, offsets):
    ox, oy, new_w, new_h = offsets
    resized = mask.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("L", (SD_SIZE, SD_SIZE), 0)
    canvas.paste(resized, (ox, oy))
    return canvas


def restore_from_sd(result_sd: Image.Image, offsets, orig_size):
    """Undo padding and restore original dimensions."""
    ox, oy, new_w, new_h = offsets
    cropped = result_sd.crop((ox, oy, ox + new_w, oy + new_h))
    return cropped.resize(orig_size, Image.LANCZOS)


def build_prompt(material: str) -> tuple[str, str]:
    if material == "plastico":
        return (
            "wrapped in shiny polypropylene plastic sheeting tied with thick nylon rope, "
            "Christo and Jeanne-Claude art installation, plastic wrapping follows the exact shape and silhouette, "
            "photorealistic photograph, natural lighting, high detail",
            "deformed shape, reshaped, different proportions, blob, ugly, blurry, cartoon, watermark"
        )
    return (
        "wrapped in white linen fabric cloth tied with thick rope, "
        "Christo and Jeanne-Claude art installation, fabric follows the exact shape and silhouette of the object, "
        "photorealistic photograph, natural lighting, high detail",
        "deformed shape, reshaped, different proportions, blob, ugly, blurry, cartoon, watermark"
    )


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

    # 1. Load and fix orientation
    contents = await file.read()
    try:
        pil = Image.open(BytesIO(contents))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen invalida")

    orig_size = pil.size

    # 2. Segment object locally with GrabCut (no API call)
    mask_orig = segment_object_grabcut(pil)

    # 3. Resize image and mask to SD canvas
    sd_img, offsets = resize_for_sd(pil)
    sd_mask = resize_mask_for_sd(mask_orig, offsets)

    # 4. Single Replicate call: SD inpainting
    prompt, negative_prompt = build_prompt(material)

    try:
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "image": pil_to_data_uri(sd_img),
                "mask": pil_to_data_uri(sd_mask),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "strength": 0.55,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en generacion: {str(e)}")

    # 5. Get result URL
    output_list = list(output) if not isinstance(output, list) else output
    if not output_list:
        raise HTTPException(status_code=500, detail="No se obtuvo resultado")

    first = output_list[0]
    result_url = first if isinstance(first, str) else getattr(first, "url", str(first))
    if not result_url.startswith("http"):
        raise HTTPException(status_code=500, detail="URL invalida")

    # 6. Composite: paste result only within mask, keep original background
    result_bytes = await download_bytes(result_url)
    result_sd = Image.open(BytesIO(result_bytes)).convert("RGB").resize((SD_SIZE, SD_SIZE), Image.LANCZOS)

    composite_sd = Image.composite(result_sd, sd_img, sd_mask)

    # 7. Restore original dimensions
    final = restore_from_sd(composite_sd, offsets, orig_size)
    result_b64 = image_to_base64(final)

    return JSONResponse({
        "result_image": result_b64,
        "width": final.width,
        "height": final.height,
    })
