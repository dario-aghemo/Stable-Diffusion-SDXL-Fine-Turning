from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from rembg import new_session, remove

try:
    SEG_SESSION = new_session("isnet-general-use")
except:
    SEG_SESSION = new_session("isnet-general-use")

WHITE_BG = (255, 255, 255)

def get_mask_with_rembg(pil_image: Image.Image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    out_bytes = remove(buffer.read(), session=SEG_SESSION)
    out_rgba = Image.open(BytesIO(out_bytes)).convert("RGBA")
    alpha = np.array(out_rgba.split()[-1])
    return alpha

def postprocess_mask(mask: np.ndarray, kernel_size=3, blur_kernel=9):
    _, mask_bin = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = (labels == largest).astype(np.uint8) * 255
    feathered = cv2.GaussianBlur(cleaned, (blur_kernel | 1, blur_kernel | 1), 0)
    return feathered.astype(np.uint8)

def apply_mask_transparent(original: Image.Image, mask: np.ndarray):
    rgba = original.convert("RGBA")
    mask_pil = Image.fromarray(mask).resize(rgba.size, Image.BILINEAR)
    rgba.putalpha(mask_pil)
    return rgba  

def center_on_transparent(obj: Image.Image, canvas_size: int):
    
    alpha = obj.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        canvas.paste(obj, ((canvas_size - obj.width)//2, (canvas_size - obj.height)//2), obj)
        return canvas

    cropped = obj.crop(bbox)
 
    side = max(cropped.width, cropped.height)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(cropped, ((side - cropped.width)//2, (side - cropped.height)//2), cropped)

    resized = square.resize((canvas_size, canvas_size), Image.LANCZOS)
    return resized

def add_natural_shadow(centered_img: Image.Image,
                       shadow_strength: int = 120,
                       shadow_blur: int = 28,
                       offset_ratio: float = 0.06,
                       height_ratio: float = 0.18):
    """
    centered_img: RGBA image already sized to output_size (square).
    shadow_strength: 0-255 alpha for shadow darkness (120 = subtle).
    shadow_blur: gaussian blur radius.
    offset_ratio: vertical offset as fraction of image height.
    height_ratio: how far to expand the mask downward before blur.
    """
    w, h = centered_img.size
    alpha = centered_img.split()[-1]

    shift_y = int(h * offset_ratio)
    expand_h = int(h * height_ratio)

    shifted = Image.new("L", (w, h), 0)
    shifted.paste(alpha, (0, shift_y))

    if expand_h > 0:
        shifted.paste(alpha, (0, shift_y + expand_h // 2))

    blurred = shifted.filter(ImageFilter.GaussianBlur(shadow_blur))
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow_color = (0, 0, 0, shadow_strength)
    shadow_layer.paste(shadow_color, (0, 0), blurred)

    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base = Image.alpha_composite(base, shadow_layer)
    base.paste(centered_img, (0, 0), centered_img)
    return base

def compose_on_white(rgba_img: Image.Image):
    w, h = rgba_img.size
    white = Image.new("RGBA", (w, h), WHITE_BG + (255,))
    composed = Image.alpha_composite(white, rgba_img)
    return composed.convert("RGB")

def process_glasses_image(image_bytes: bytes, output_size: int = 1024):
    
    original = Image.open(BytesIO(image_bytes)).convert("RGBA")

    temp_bg = Image.new("RGBA", original.size, (240, 240, 240, 255))
    pre = Image.alpha_composite(temp_bg, original)

    mask_raw = get_mask_with_rembg(pre.convert("RGB"))
    mask_pp = postprocess_mask(mask_raw)

    transparent_obj = apply_mask_transparent(original.convert("RGB"), mask_pp)
    centered = center_on_transparent(transparent_obj, output_size)

    shadowed = add_natural_shadow(centered,
                                  shadow_strength=120,  
                                  shadow_blur=28,
                                  offset_ratio=0.06,
                                  height_ratio=0.18)

    final_rgb = compose_on_white(shadowed)
    final_rgb = ImageEnhance.Sharpness(final_rgb).enhance(1.15)

    buf = BytesIO()
    final_rgb.save(buf, format="PNG", optimize=True)
    return buf.getvalue()



# --------------------**********************------------------------

#  SHADOW SHORTER FRAME FIXED

# -------------------**********************--------------------------



# from io import BytesIO
# import numpy as np
# from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
# import cv2
# from rembg import new_session, remove

# try:
#     SEG_SESSION = new_session("isnet-general-use")
# except:
#     SEG_SESSION = new_session("isnet-general-use")

# WHITE_BG = (255, 255, 255)

# def get_mask_with_rembg(pil_image: Image.Image):
#     buffer = BytesIO()
#     pil_image.save(buffer, format="PNG")
#     buffer.seek(0)
#     out_bytes = remove(buffer.read(), session=SEG_SESSION)
#     out_rgba = Image.open(BytesIO(out_bytes)).convert("RGBA")
#     alpha = np.array(out_rgba.split()[-1])
#     return alpha

# def postprocess_mask(mask: np.ndarray, kernel_size=3, blur_kernel=9):
#     _, mask_bin = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
#     cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
#     cleaned = cv2.dilate(cleaned, kernel)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
#     if num_labels > 1:
#         largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         cleaned = (labels == largest).astype(np.uint8) * 255
#     feathered = cv2.GaussianBlur(cleaned, (blur_kernel | 1, blur_kernel | 1), 0)
#     return feathered.astype(np.uint8)

# def apply_mask_transparent(original: Image.Image, mask: np.ndarray):
#     rgba = original.convert("RGBA")
#     mask_pil = Image.fromarray(mask).resize(rgba.size, Image.BILINEAR)
#     rgba.putalpha(mask_pil)
#     return rgba

# def center_on_transparent(obj: Image.Image, canvas_size: int):
#     alpha = obj.split()[-1]
#     bbox = alpha.getbbox()
#     if bbox is None:
#         canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
#         canvas.paste(obj, ((canvas_size - obj.width) // 2, (canvas_size - obj.height) // 2), obj)
#         return canvas
#     cropped = obj.crop(bbox)
#     side = max(cropped.width, cropped.height)
#     square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
#     square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2), cropped)
#     resized = square.resize((canvas_size, canvas_size), Image.LANCZOS)
#     return resized

# def add_shadow_under_lenses_fixed(img: Image.Image,
#                                   shadow_strength: int = 150,
#                                   blur_radius: int = 12,
#                                   vertical_ratio: float = 0.0,
#                                   height_ratio: float = 0.025,
#                                   width_ratio: float = 0.18):
#     """
#     Adds subtle shadow touching the bottom of lenses, guaranteed to appear.
#     """
#     w, h = img.size
#     alpha = np.array(img.split()[-1])

#     ys, xs = np.where(alpha > 10)
#     if len(xs) == 0 or len(ys) == 0:
#         return img

#     x_min, x_max = np.min(xs), np.max(xs)
#     y_max = np.max(ys)

#     center_left = x_min + (x_max - x_min)//4
#     center_right = x_min + 3*(x_max - x_min)//4
#     base_y = y_max

#     sh_h = int(h * height_ratio)
#     sh_w = int(w * width_ratio)
#     offset_y = int(h * vertical_ratio)

#     shadow_mask = Image.new("L", (w, h), 0)
#     draw = ImageDraw.Draw(shadow_mask)

#     draw.ellipse([center_left - sh_w//2, base_y + offset_y,
#                   center_left + sh_w//2, base_y + offset_y + sh_h],
#                  fill=shadow_strength)
#     draw.ellipse([center_right - sh_w//2, base_y + offset_y,
#                   center_right + sh_w//2, base_y + offset_y + sh_h],
#                  fill=shadow_strength)

#     shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(blur_radius))
#     shadow_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
#     shadow_rgba.paste((0,0,0,255), (0,0), shadow_mask)

#     base = Image.new("RGBA", (w,h), (0,0,0,0))
#     base = Image.alpha_composite(base, shadow_rgba)
#     base.paste(img, (0,0), img)
#     return base

# def compose_on_white(rgba_img: Image.Image):
#     w,h = rgba_img.size
#     white = Image.new("RGBA", (w,h), WHITE_BG + (255,))
#     return Image.alpha_composite(white, rgba_img).convert("RGB")

# def process_glasses_image(image_bytes: bytes, output_size: int = 1024):
#     original = Image.open(BytesIO(image_bytes)).convert("RGBA")
#     temp_bg = Image.new("RGBA", original.size, (240,240,240,255))
#     pre = Image.alpha_composite(temp_bg, original)

#     mask_raw = get_mask_with_rembg(pre.convert("RGB"))
#     mask_pp = postprocess_mask(mask_raw)
#     transparent_obj = apply_mask_transparent(original.convert("RGB"), mask_pp)
#     centered = center_on_transparent(transparent_obj, output_size)

#     shadowed = add_shadow_under_lenses_fixed(centered,
#                                              shadow_strength=150,
#                                              blur_radius=12,
#                                              vertical_ratio=0.0,
#                                              height_ratio=0.025,
#                                              width_ratio=0.18)

#     final_img = compose_on_white(shadowed)
#     final_img = ImageEnhance.Sharpness(final_img).enhance(1.15)
#     buf = BytesIO()
#     final_img.save(buf, format="PNG", optimize=True)
#     return buf.getvalue()


