# from rembg import remove
# from PIL import Image, ImageOps
# import io
# import base64

# BACKGROUND_COLOR = (247, 247, 247, 255)  # #F7F7F7

# def make_square_and_center(img: Image.Image, size=1024, bg_color=BACKGROUND_COLOR):
#     img = img.convert("RGBA")
#     bbox = img.getbbox()
#     if bbox:
#         img = img.crop(bbox)
#     max_side = max(img.width, img.height)
#     scale = size / max_side
#     new_w = int(img.width * scale)
#     new_h = int(img.height * scale)
#     img = img.resize((new_w, new_h), Image.LANCZOS)
#     background = Image.new("RGBA", (size, size), bg_color)
#     offset = ((size - new_w) // 2, (size - new_h) // 2)
#     background.paste(img, offset, img)
#     return background

# def remove_bg_and_prepare(image_bytes: bytes, output_size=1024):
#     # Use alpha matting for better fine edges
#     try:
#         no_bg_bytes = remove(
#             image_bytes,
#             alpha_matting=True,
#             alpha_matting_foreground_threshold=240,
#             alpha_matting_background_threshold=10,
#             alpha_matting_erode_size=1
#         )
#     except Exception as e:
#         raise RuntimeError(f"Background removal failed: {e}")
    
#     img = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
#     final = make_square_and_center(img, size=output_size)
#     buffer = io.BytesIO()
#     final.save(buffer, format="PNG")
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")
# cleanup.py
# Improved product cleanup pipeline using ISNet (isnet-general-use) segmentation model.
# Behavior:
#  - Always runs segmentation (even for PNGs with alpha)
#  - Robust mask postprocessing (morphology, blur/feather)
#  - Centers the glasses, makes a square canvas, fills white background (#FFFFFF)
#  - Returns a PIL.Image (RGB) ready to save or encode to base64


from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from rembg import new_session, remove

try:
    SEG_SESSION = new_session("isnet-general-use")
except:
    SEG_SESSION = new_session("isnet-general-use")

WHITE_BG = (255, 255, 255)

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def get_mask_with_rembg(pil_image: Image.Image):
    """Generate raw alpha mask from ISNet model."""
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    out_bytes = remove(buffer.read(), session=SEG_SESSION)
    out_rgba = Image.open(BytesIO(out_bytes)).convert("RGBA")
    mask = np.array(out_rgba.split()[-1])  # alpha channel

    return mask

def postprocess_mask(mask: np.ndarray, kernel_size=3, blur_kernel=9):
    """Clean mask edges while preserving thin frames."""
    _, mask_bin = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = (labels == largest_label).astype(np.uint8) * 255

    feathered = cv2.GaussianBlur(cleaned, (blur_kernel | 1, blur_kernel | 1), 0)
    return feathered.astype(np.uint8)

def apply_mask(original: Image.Image, mask: np.ndarray):
    """Apply mask to RGB original and composite on white background."""
    rgba = original.convert("RGBA")
    mask_pil = Image.fromarray(mask).resize(rgba.size, Image.BILINEAR)

    rgba.putalpha(mask_pil)

    bg = Image.new("RGB", rgba.size, WHITE_BG)
    result = Image.alpha_composite(bg.convert("RGBA"), rgba).convert("RGB")
    return result

def center_and_square(img: Image.Image, canvas_size=1024):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(bw)
    if coords is None:
        canvas = Image.new("RGB", (canvas_size, canvas_size), WHITE_BG)
        canvas.paste(img, ((canvas_size - img.width)//2, (canvas_size - img.height)//2))
        return canvas

    x, y, w, h = cv2.boundingRect(coords)
    obj = img.crop((x, y, x+w, y+h))

    side = max(w, h)
    canvas = Image.new("RGB", (side, side), WHITE_BG)
    canvas.paste(obj, ((side - w)//2, (side - h)//2))

    return canvas.resize((canvas_size, canvas_size), Image.LANCZOS)

def process_glasses_image(image_bytes: bytes, output_size=1024):
    """Main processing pipeline exposed to FastAPI."""
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")

    bg = Image.new("RGBA", img.size, (240, 240, 240, 255))
    img_pre = Image.alpha_composite(bg, img)

    mask_raw = get_mask_with_rembg(img_pre.convert("RGB"))
    mask_pp = postprocess_mask(mask_raw)

    original_rgb = Image.open(BytesIO(image_bytes)).convert("RGB")
    masked = apply_mask(original_rgb, mask_pp)

    final = center_and_square(masked, canvas_size=output_size)

    final = ImageEnhance.Sharpness(final).enhance(1.2)

    buf = BytesIO()
    final.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
