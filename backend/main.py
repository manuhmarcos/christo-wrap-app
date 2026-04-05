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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
SD_SIZE = 512


# ── Image utilities ──────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_data_uri(img: Image.Image, fmt="PNG") -> str:
    return f"data:image/png;base64,{image_to_base64(img, fmt)}"


async def download_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


# ── Segmentation ─────────────────────────────────────────────────────────────

def segment_grabcut(img_pil: Image.Image) -> np.ndarray:
    """
    Segments the main object using GrabCut.
    Returns a uint8 mask (255=object, 0=background), same size as img_pil.
    """
    arr = np.array(img_pil.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    mx, my = int(w * 0.10), int(h * 0.06)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 10, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        fg = np.zeros((h, w), np.uint8)
        cv2.ellipse(fg, (w // 2, h // 2), (int(w * 0.35), int(h * 0.42)), 0, 0, 360, 255, -1)

    # Clean up
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open,  iterations=1)

    # Keep largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = np.where(labels == best, 255, 0).astype(np.uint8)

    return fg


def erode_mask(mask_np: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.erode(mask_np, k, iterations=1)


def smooth_mask(mask_np: np.ndarray, blur: int = 9) -> Image.Image:
    blurred = cv2.GaussianBlur(mask_np, (blur * 2 + 1, blur * 2 + 1), 0)
    return Image.fromarray(blurred, mode="L")


# ── Resize / restore ─────────────────────────────────────────────────────────

def resize_for_sd(img: Image.Image):
    w, h = img.size
    ratio = SD_SIZE / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (SD_SIZE, SD_SIZE), 0)
    ox, oy = (SD_SIZE - nw) // 2, (SD_SIZE - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, (ox, oy, nw, nh)


def mask_for_sd(mask_np: np.ndarray, offsets, blur: int = 7) -> Image.Image:
    ox, oy, nw, nh = offsets
    m = Image.fromarray(mask_np, mode="L").resize((nw, nh), Image.NEAREST)
    canvas = Image.new("L", (SD_SIZE, SD_SIZE), 0)
    canvas.paste(m, (ox, oy))
    return smooth_mask(np.array(canvas), blur)


def restore(img_sd: Image.Image, offsets, orig_size) -> Image.Image:
    ox, oy, nw, nh = offsets
    return img_sd.crop((ox, oy, ox + nw, oy + nh)).resize(orig_size, Image.LANCZOS)


# ── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(material: str):
    if material == "plastico":
        pos = (
            "object wrapped in shiny silver polypropylene plastic sheeting, "
            "thick nylon rope tied around it in Christo and Jeanne-Claude style, "
            "plastic wrap with realistic reflections and creases, "
            "professional art installation photography, dramatic side lighting, "
            "the wrapped bundle shape clearly shows the object underneath"
        )
    else:
        pos = (
            "object wrapped in off-white linen canvas fabric, "
            "thick nautical rope tied around it in Christo and Jeanne-Claude style, "
            "fabric with realistic folds draping and wrinkles following the object shape, "
            "professional art installation photography, dramatic side lighting, "
            "the wrapped bundle shape clearly shows the object underneath"
        )
    neg = (
        "naked object, unwrapped, no fabric, no cloth, no rope, "
        "deformed, melted, blob shape, ugly, blurry, cartoon, painting, text, watermark, "
        "different background, floating"
    )
    return pos, neg


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/wrap")
async def wrap_object(file: UploadFile = File(...), material: str = "tela"):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(500, "REPLICATE_API_TOKEN no configurado")
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

    # 1. Load + fix EXIF rotation
    contents = await file.read()
    try:
        pil = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Imagen invalida")
    orig_size = pil.size

    # 2. Segment object (GrabCut, no API call)
    fg_mask = segment_grabcut(pil)

    # 3. Resize to SD canvas
    sd_img, offsets = resize_for_sd(pil)

    # 4. Build inpainting mask = full object area (smooth edges for blending)
    fg_sd = np.array(
        Image.fromarray(fg_mask).resize((offsets[2], offsets[3]), Image.NEAREST)
    )
    ox, oy, nw, nh = offsets
    fg_canvas = np.zeros((SD_SIZE, SD_SIZE), np.uint8)
    fg_canvas[oy:oy+nh, ox:ox+nw] = fg_sd

    inpaint_mask = smooth_mask(fg_canvas, blur=5)

    # 5. Single Replicate call
    # strength=0.82: strong enough to generate real fabric texture + folds,
    # low enough that SD still uses the original structure as reference → shape preserved
    prompt, neg = build_prompt(material)
    try:
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            input={
                "image":           pil_to_data_uri(sd_img),
                "mask":            pil_to_data_uri(inpaint_mask),
                "prompt":          prompt,
                "negative_prompt": neg,
                "num_inference_steps": 50,
                "guidance_scale":  9.0,
                "strength":        0.82,
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Error en generacion: {e}")

    # 6. Get result
    items = list(output) if not isinstance(output, list) else output
    if not items:
        raise HTTPException(500, "Sin resultado")
    first = items[0]
    url = first if isinstance(first, str) else getattr(first, "url", str(first))
    if not url.startswith("http"):
        raise HTTPException(500, "URL invalida")

    result_bytes = await download_bytes(url)
    result_sd = Image.open(BytesIO(result_bytes)).convert("RGB").resize((SD_SIZE, SD_SIZE))

    # 7. Composite: fabric inside mask, original background outside
    composite_sd = Image.composite(result_sd, sd_img, inpaint_mask)

    # 8. Restore original dimensions
    final = restore(composite_sd, offsets, orig_size)
    return JSONResponse({"result_image": image_to_base64(final), "width": final.width, "height": final.height})
