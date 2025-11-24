# from fastapi import FastAPI, UploadFile, File, HTTPException
# from cleanup import remove_bg_and_prepare
# from pydantic import BaseModel

# app = FastAPI(title="Product Image Cleanup API", version="1.0")

# class HealthResponse(BaseModel):
#     status: str

# @app.get("/")
# async def root():
#     return {"message": "Glasses Cleanup API is running. Use /docs for API documentation."}

# @app.get("/health", response_model=HealthResponse)
# async def health():
#     return {"status": "ok"}

# @app.post("/cleanup")
# async def cleanup(file: UploadFile = File(...)):
#     if file.content_type.split('/')[0] != 'image':
#         raise HTTPException(status_code=400, detail="Uploaded file must be an image")
#     image_bytes = await file.read()
#     try:
#         encoded_png = remove_bg_and_prepare(image_bytes, output_size=1024)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     return {"status": "success", "image_base64": encoded_png}

import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from cleanup import process_glasses_image
from pydantic import BaseModel

app = FastAPI(
    title="Product Image Cleanup API",
    version="1.1",
    description="Glasses Cleanup API with improved background removal and transparency handling."
)

class HealthResponse(BaseModel):
    status: str

@app.get("/")
async def root():
    return {"message": "Glasses Cleanup API is running. Use /docs for API documentation."}

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}

@app.post("/cleanup")
async def cleanup(file: UploadFile = File(...), output_size: int = 1024):
    """
    Upload an image to clean up background and prepare for product images.
    Returns the processed PNG as Base64 string in JSON.
    """
    if not file.content_type or file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        png_bytes = process_glasses_image(image_bytes, output_size=output_size)

        encoded_png = base64.b64encode(png_bytes).decode("utf-8")

    except Exception as e:
        raise HTTPException(500, f"Image processing error: {str(e)}")

    return {
        "status": "success",
        "image_base64": encoded_png
    }
