# Milestone 1 - Product Image Cleanup API

This project contains the Milestone 1 deliverables:
- FastAPI backend
- Product image cleanup pipeline (background removal, square crop, centering, background color, base64 output)
- Dockerfile and requirements.txt for deployment

Run locally (Windows):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Endpoints:
- POST /cleanup
  - Form field: file (image)
  - Returns JSON: { "status": "success", "image_base64": "<base64 PNG>" }

Notes:
- This implementation uses `rembg` for background removal (U^2-Net).
- Optional upscaling via Real-ESRGAN is not included by default (commented instructions in code).
