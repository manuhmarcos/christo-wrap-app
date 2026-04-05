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

# SD inpainting works best at these sizes
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


async def download_image(url: str) -> Image.Image:
    data = await download_bytes(url)
    return Image.open(BytesIO(data)).convert("RGB")


def extract_canny_edges(img: Image.Image) -> Image.Image:
    """Extract canny edges from image — used by ControlNet to preserve object shape."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    # Convert single channel to RGB (ControlNet expects RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def resize_to_sd(img: Image.Image) -> Image.Image:
    """Resize keeping aspect ratio so longest side = SD_SIZE, then pad to square."""
    w, h = img.size
    ratio = SD_SIZE / max(w, h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    # Pad to SD_SIZE x SD_SIZE with black
    padded = Image.new("RGB", (SD_SIZE, SD_SIZE), (0, 0, 0))
    offset_x = (SD_SIZE - new_w) // 2
    offset_y = (SD_SIZE - new_h) // 2
    padded.paste(resized, (offset_x, offset_y))
    return padded, (offset_x, offset_y, new_w, new_h)


def resize_mask_to_sd(mask: Image.Image, offsets: tuple) -> Image.Image:
    offset_x, offset_y, new_w, new_h = offsets
    resized = mask.resize((new_w, new_h), Image.NEAREST)
    padded = Image.new("L", (SD_SIZE, SD_SIZE), 0)
    padded.paste(resized, (offset_x, offset_y))
    return padded


def crop_and_resize_result(result: Image.Image, offsets: tuple, orig_size: tuple) -> Image.Image:
    """Undo padding and restore original dimensions."""
    offset_x, offset_y, new_w, new_h = offsets
    cropped = result.crop((offset_x, offset_y, offset_x + new_w, offset_y + new_h))
    return cropped.resize(orig_size, Image.LANCZOS)


def create_center_mask(width: int, height: int, coverage: float = 0.55) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    margin_x = int(width * (1 - coverage) / 2)
    margin_y = int(height * (1 - coverage) / 2)
    draw.ellipse([margin_x, margin_y, width - margin_x, height - margin_y], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
    return mask


def dilate_mask(mask_arr: np.ndarray, pixels: int = 8) -> Image.Image:
    mask_img = Image.fromarray((mask_arr > 128).astype(np.uint8) * 255, mode="L")
    for _ in range(max(1, pixels // 5)):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(7))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
    return mask_img


def select_best_mask(masks: list, width: int, height: int) -> np.ndarray | None:
    cx, cy = width / 2, height / 2
    best_score = -1
    best_mask = None
    for arr in masks:
        area = np.sum(arr > 128)
        if area == 0:
            continue
        ys, xs = np.where(arr > 128)
        mcx, mcy = np.mean(xs), np.mean(ys)
        dist = np.sqrt((mcx - cx) ** 2 + (mcy - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        centrality = 1 - (dist / (max_dist + 1e-6))
        score = 0.6 * (area / (width * height)) + 0.4 * centrality
        if score > best_score:
            best_score = score
            best_mask = arr
    return best_mask


def build_prompt(material: str) -> tuple[str, str]:
    if material == "plastico":
        return (
            "wrapped in shiny transparent polypropylene plastic sheet tied with rope, "
            "Christo and Jeanne-Claude art style, same object shape preserved under plastic, "
            "plastic follows exact contours, realistic photography, studio lighting",
            "deformed, reshaped, different proportions, blob, ugly, blurry, cartoon, text"
        )
    return (
        "wrapped in white linen cloth fabric tied with rope, "
        "Christo and Jeanne-Claude art style, same object shape preserved under fabric, "
        "cloth follows exact contours and silhouette, realistic photography, studio lighting",
        "deformed, reshaped, different proportions, blob, ugly, blurry, cartoon, text"
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

    # 1. Load image and fix EXIF rotation
    contents = await file.read()
    try:
        original_image = Image.open(BytesIO(contents))
        original_image = ImageOps.exif_transpose(original_image)
        original_image = original_image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen invalida")

    orig_size = original_image.size  # Save for final restore
    orig_w, orig_h = orig_size

    # 2. Resize to SD_SIZE with padding (preserving aspect ratio)
    sd_image, offsets = resize_to_sd(original_image)
    _, _, new_w, new_h = offsets

    # 3. Extract Canny edges (same size as sd_image)
    canny_image = extract_canny_edges(sd_image)

    # 4. Run SAM segmentation on sd_image
    mask_arr = None
    try:
        sam_output = replicate.run(
            "meta/sam-2",
            input={
                "image": pil_to_data_uri(sd_image),
                "use_m2m": True,
                "points_per_side": 32,
                "pred_iou_thresh": 0.86,
                "stability_score_thresh": 0.92,
            }
        )
        masks = []
        for item in (list(sam_output) if not isinstance(sam_output, list) else sam_output):
            url = item if isinstance(item, str) else getattr(item, "url", str(item))
            if url and url.startswith("http"):
                try:
                    mask_img = await download_image(url)
                    arr = np.array(mask_img.resize((SD_SIZE, SD_SIZE)).convert("L"))
                    if np.sum(arr > 128) > 0:
                        masks.append(arr)
                except Exception:
                    continue
        if masks:
            mask_arr = select_best_mask(masks, SD_SIZE, SD_SIZE)
    except Exception:
        mask_arr = None

    # 5. Build mask
    if mask_arr is not None:
        mask_sd = dilate_mask(mask_arr, pixels=8)
    else:
        mask_sd = create_center_mask(SD_SIZE, SD_SIZE, coverage=0.55)

    # 6. Run ControlNet + Inpainting via SDXL
    # We use controlnet-canny to preserve shape + inpainting mask for fabric
    prompt, negative_prompt = build_prompt(material)

    try:
        # Use controlnet with canny to preserve structure
        output = replicate.run(
            "rossjillian/controlnet:795433b19458d0f4fa172a7ccf93178d2adb1cb8ab2ad6c8fdc33fdbcd49f477",
            input={
                "image": pil_to_data_uri(canny_image),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "structure": "canny",
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "image_resolution": SD_SIZE,
            }
        )
    except Exception as e1:
        # Fallback to plain inpainting if controlnet fails
        try:
            output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "image": pil_to_data_uri(sd_image),
                    "mask": pil_to_data_uri(mask_sd),
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "strength": 0.50,
                }
            )
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Error en generacion: {str(e2)}")

    # 7. Get result URL
    output_list = list(output) if not isinstance(output, list) else output
    if not output_list:
        raise HTTPException(status_code=500, detail="No se obtuvo imagen de resultado")

    first = output_list[0]
    result_url = first if isinstance(first, str) else getattr(first, "url", str(first))
    if not result_url.startswith("http"):
        raise HTTPException(status_code=500, detail="URL de resultado invalida")

    # 8. Download result and restore original dimensions
    result_bytes = await download_bytes(result_url)
    result_sd = Image.open(BytesIO(result_bytes)).convert("RGB")

    # Composite: only apply result within mask, keep background from original
    result_sd_resized = result_sd.resize((SD_SIZE, SD_SIZE), Image.LANCZOS)
    mask_rgba = mask_sd.convert("L")
    composite = Image.composite(result_sd_resized, sd_image, mask_rgba)

    # Restore original proportions (undo padding, restore original size)
    final_image = crop_and_resize_result(composite, offsets, orig_size)

    result_b64 = image_to_base64(final_image)

    return JSONResponse({
        "result_image": result_b64,
        "width": final_image.width,
        "height": final_image.height,
    })
