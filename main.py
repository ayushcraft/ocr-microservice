"""
Production FastAPI + PaddleOCR 2.8.1 (PP-OCRv4) OCR Microservice
Optimized for small text detection with advanced preprocessing pipeline.
Returns ALL variant results (original, thresholded, sharpened, blackhat, CLAHE) per file.
"""

import os
import cv2
import uuid
import time
import magic
import string
import asyncio
import logging
import numpy as np
import json
import contextvars

from typing import List
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from paddleocr import PaddleOCR

# ================= CONFIG =================
MAX_FILES = 20
MAX_FILE_SIZE_MB = 20
MAX_TOTAL_UPLOAD_MB = 200
MAX_PIXELS = 25_000_000

ALLOWED_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

# ================= LOGGING =================
request_id_ctx_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx_var.get()
        return True


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        ignored_keys = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'message', 'name', 'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'thread', 'threadName', 'request_id', 'msg', 'taskName'
        }

        for k, v in record.__dict__.items():
            if k not in ignored_keys:
                log_data[k] = v

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        def default_serializer(obj):
            try:
                return str(obj)
            except Exception:
                return repr(obj)

        return json.dumps(log_data, default=default_serializer)


logger = logging.getLogger("ocr_service")
logger.setLevel(logging.INFO)
logger.handlers.clear()

handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
handler.addFilter(ContextFilter())
logger.addHandler(handler)

# ================= APP =================
app = FastAPI(title="Medicine OCR Service")

# ================= MODEL =================
ocr_engine = None


def load_ocr():
    """Initialize PaddleOCR with advanced parameters for small text detection."""
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False,
        use_gpu=False,  # Set True in production with GPU
        # Explicitly load server models (auto-download if None)
        det_model_dir=None,  # PP-OCRv4_server_det
        rec_model_dir=None,  # PP-OCRv4_server_rec (SVTR-based)
        # Small text optimization
        det_db_box_thresh=0.3,
        drop_score=0.3,
        det_limit_side_len=1280,  # Higher resolution for small text
    )

#hello




# ================= MIDDLEWARE =================
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_ctx_var.set(req_id)
        try:
            response = await call_next(request)
            return response
        finally:
            request_id_ctx_var.reset(token)


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout=60):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error("Request timed out", extra={"timeout_seconds": self.timeout})
            return JSONResponse(status_code=408, content={"success": False, "error": "request_timeout"})


app.add_middleware(RequestIDMiddleware)
app.add_middleware(TimeoutMiddleware, timeout=60)


# ================= STARTUP =================
@app.on_event("startup")
async def startup():
    global ocr_engine
    logger.info("Application startup initiated")
    logger.info("Loading PP-OCRv4 model into memory...")

    ocr_engine = load_ocr()
    logger.info("PP-OCRv4 model loaded successfully")

    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    logger.info("Warming up model with dummy prediction...")
    try:
        await asyncio.to_thread(ocr_engine.ocr, dummy, cls=True)
        logger.info("Model warm-up complete")
    except Exception as exc:
        logger.warning("Model warm-up failed (non-critical)", extra={"error": str(exc)})

    logger.info("Application startup completed. Service is ready.")


# ================= UTILITIES =================
mime_magic = magic.Magic(mime=True)


def sanitize_filename(name: str) -> str:
    return os.path.basename(name or "unknown")


def validate_signature(contents: bytes):
    mime = mime_magic.from_buffer(contents)
    if mime not in ALLOWED_MIMES:
        raise ValueError("invalid_file_signature")


def decode_image(contents: bytes):
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid_image")
    h, w = image.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError("image_dimensions_too_large")
    return image


def resize_image(image, max_width=1500):
    """Resize image down to a max width to prevent OOM errors during 2x upscale."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


# ================= ADVANCED PREPROCESSING PIPELINE =================
def full_preprocess_pipeline(image):
    """
    Exact sequence:
    1. 2x upscale
    2. Glare removal (via inpainting)
    3. Denoise
    4. CLAHE
    5. Perspective correction (via deskewing)
    """
    # 1. 2x Upscale (Interpolation CUBIC is best for upscaling)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. Glare removal (via inpainting)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, glare_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_mask = cv2.dilate(glare_mask, np.ones((3, 3), np.uint8), iterations=2)
    if cv2.countNonZero(glare_mask) > 0:
        image = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)

    # 3. Denoise
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 5. Perspective correction (via deskewing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 10:  # Avoid errors on empty images
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 1.0:  # Only correct if skew is noticeable
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return image


def generate_variants(image):
    """
    Generate 5 distinct variants for Multi-pass OCR:
    1. Original (from pipeline)
    2. Thresholded (adaptive thresholding)
    3. Sharpened (unsharp mask)
    4. Edge-enhanced (Morphological black-hat transformation)
    5. CLAHE (Stronger CLAHE variant)
    """
    variants = []

    # 1. Original
    variants.append(("original", image.copy()))

    # 2. Thresholded (Adaptive Thresholding)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(("thresholded", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)))

    # 3. Sharpened (Unsharp Mask)
    blur = cv2.GaussianBlur(image, (0, 0), 2)
    sharpened = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    variants.append(("sharpened", sharpened))

    # 4. Edge-enhanced / Morphological black-hat transformation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    variants.append(("blackhat", cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)))

    # 5. CLAHE (Distinct variant with stronger clip limit)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe_strong.apply(l)
    clahe_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    variants.append(("clahe", clahe_img))

    return variants


def preprocess(contents):
    validate_signature(contents)
    image = decode_image(contents)
    original_h, original_w = image.shape[:2]

    # Limit base size to prevent out-of-memory errors during 2x upscale
    image = resize_image(image, max_width=1500)

    # Apply advanced pipeline
    image = full_preprocess_pipeline(image)

    return image, original_w, original_h


# ================= OCR =================
def run_ocr_multi_pass(image):
    variants = generate_variants(image)
    all_results = []
    for pass_name, img in variants:
        start_pass = time.time()
        try:
            result = ocr_engine.ocr(img, cls=True)
            duration_pass = time.time() - start_pass
            logger.info("OCR pass completed",
                        extra={"pass_variant": pass_name, "duration_ms": round(duration_pass * 1000, 2)})
            all_results.append((pass_name, result))
        except Exception as exc:
            duration_pass = time.time() - start_pass
            logger.warning("OCR pass failed",
                           extra={"pass_variant": pass_name, "duration_ms": round(duration_pass * 1000, 2),
                                  "error": str(exc)})
    return all_results


def _extract_elements_from_batch(batch):
    """Extract sorted elements from a single PaddleOCR 2.x batch result."""
    elements = []
    if not batch or not batch[0]:
        return elements

    for line in batch[0]:
        try:
            poly = line[0]
            text = line[1][0].strip()
            score = float(line[1][1])

            # Adjusted drop_score filter based on model initialization
            if score < 0.3 or not text:
                continue

            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]

            elements.append({
                "id": None,
                "text": text,
                "confidence": round(score, 3),
                "bounding_box": {
                    "x": round(float(min(xs)), 2),
                    "y": round(float(min(ys)), 2),
                    "width": round(float(max(xs) - min(xs)), 2),
                    "height": round(float(max(ys) - min(ys)), 2)
                }
            })
        except Exception:
            continue

    elements.sort(key=lambda e: (e["bounding_box"]["y"], e["bounding_box"]["x"]))
    return elements


def process_results(pass_results, width, height, filename):
    """Process ALL variant results and return each one separately."""
    variant_results = []

    for pass_name, batch in pass_results:
        elements = _extract_elements_from_batch(batch)
        full_text = " ".join(e["text"] for e in elements)
        avg_conf = round(
            sum(e["confidence"] for e in elements) / len(elements), 3
        ) if elements else None

        variant_results.append({
            "variant_name": pass_name,
            "full_text_combined": full_text if full_text else None,
            "average_confidence": avg_conf,
            "element_count": len(elements),
            "elements": elements
        })

    has_any_detections = any(v["element_count"] > 0 for v in variant_results)

    return {
        "filename": filename,
        "status": "success" if has_any_detections else "no_detections",
        "image_width": width,
        "image_height": height,
        "variants": variant_results
    }


# ================= ENDPOINTS =================
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"ready": ocr_engine is not None}


@app.post("/ocr")
async def ocr_endpoint(request: Request, files: List[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        logger.warning("Request rejected: too many files", extra={"file_count": len(files), "max_allowed": MAX_FILES})
        return JSONResponse(status_code=400, content={"success": False, "error": "too_many_files"})

    filenames = [sanitize_filename(f.filename) for f in files]
    logger.info("Request data received", extra={"file_count": len(files), "filenames": filenames})

    file_results = []
    total_bytes = 0
    start = time.time()

    for file in files:
        safe_name = sanitize_filename(file.filename)
        try:
            contents = await file.read()
            total_bytes += len(contents)

            if total_bytes > MAX_TOTAL_UPLOAD_MB * 1024 * 1024:
                raise ValueError("total_upload_limit_exceeded")
            if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError("file_too_large")

            start_pre = time.time()
            image, w, h = await asyncio.to_thread(preprocess, contents)
            duration_pre = time.time() - start_pre
            logger.info("Preprocess completed",
                        extra={"image_filename": safe_name, "duration_ms": round(duration_pre * 1000, 2),
                               "original_width": w, "original_height": h})

            start_ocr = time.time()
            pass_results = await asyncio.to_thread(run_ocr_multi_pass, image)
            duration_ocr = time.time() - start_ocr
            logger.info("OCR completed",
                        extra={"image_filename": safe_name, "duration_ms": round(duration_ocr * 1000, 2)})

            start_post = time.time()
            processed = await asyncio.to_thread(process_results, pass_results, w, h, safe_name)
            duration_post = time.time() - start_post

            total_det = sum(v["element_count"] for v in processed["variants"])
            logger.info("Postprocess completed",
                        extra={"image_filename": safe_name, "duration_ms": round(duration_post * 1000, 2),
                               "detections_count": total_det})

            file_results.append(processed)

        except Exception as exc:
            logger.exception("File processing failed", extra={"image_filename": safe_name})
            file_results.append({
                "filename": safe_name,
                "status": "error",
                "image_width": 0,
                "image_height": 0,
                "variants": []
            })

    processing_time = round(time.time() - start, 2)
    success_count = sum(1 for r in file_results if r.get("status") == "success")
    fail_count = len(file_results) - success_count

    logger.info("Request summary",
                extra={"total_files": len(files), "success_count": success_count, "fail_count": fail_count,
                       "total_duration_s": processing_time, "total_bytes_uploaded": total_bytes})

    return {
        "success": fail_count == 0,
        "processing_time": processing_time,
        "total_files": len(files),
        "successful": success_count,
        "failed": fail_count,
        "results": file_results
    }