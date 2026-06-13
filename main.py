# """
# Production FastAPI + PaddleOCR 3.x (PP-OCRv5) OCR Microservice
# Skeleton full implementation with:
# - Global model initialization
# - Health/readiness endpoints
# - Security validation
# - Multi-pass preprocessing
# - Threaded OCR execution
# - Batch processing
# - Structured logging
# """
#
# import os
# import cv2           # OpenCV library for image processing
# import uuid          # For generating unique request IDs
# import time          # For measuring processing time
# import magic         # Python-magic library for file type detection
# import string        # String utilities (not used here, but available)
# import asyncio       # Async I/O support for concurrent operations
# import logging       # Logging framework for structured logs
# import numpy as np   # Numerical computing library, essential for image arrays
#
# from typing import List
# from fastapi import FastAPI, UploadFile, File, Request
# from fastapi.responses import JSONResponse
# from starlette.middleware.base import BaseHTTPMiddleware
#
# from paddleocr import PaddleOCR  # PaddlePaddle's OCR toolkit
#
# # ================= CONFIG =================
#
# # Maximum number of files that can be uploaded in a single request
# MAX_FILES = 20
#
# # Maximum size for a single file (in megabytes)
# MAX_FILE_SIZE_MB = 20
#
# # Maximum total size for all files combined in one request (in megabytes)
# MAX_TOTAL_UPLOAD_MB = 200
#
# # Maximum number of pixels allowed in an image (width × height)
# # This prevents processing extremely large images that would consume too much memory
# # Example: 5000×5000 = 25,000,000 pixels
# MAX_PIXELS = 25_000_000
#
# # Set of allowed MIME types (file formats)
# # Only these image types will be accepted for security reasons
# ALLOWED_MIMES = {
#     "image/jpeg",   # JPEG/JPG format
#     "image/png",    # PNG format (supports transparency)
#     "image/webp",   # WebP format (modern, efficient compression)
# }
#
# # ================= LOGGING =================
#
# # Configure the logging system
# # level=logging.INFO: Show INFO, WARNING, ERROR, and CRITICAL messages
# # format: Defines how log messages appear
# #   %(asctime)s - Timestamp when the log was created
# #   %(levelname)s - Severity level (INFO, ERROR, etc.)
# #   %(name)s - Logger name (here: "ocr_service")
# #   %(message)s - The actual log message
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s %(message)s"
# )
#
# # Create a logger instance specific to this service
# # Using a named logger helps identify which part of the application generated the log
# logger = logging.getLogger("ocr_service")
#
# # ================= APP =================
#
# # Create the FastAPI application instance
# # title: Name that appears in the auto-generated API documentation (Swagger UI)
# app = FastAPI(title="Medicine OCR Service")
#
# # ================= MODEL =================
#
# # Global variable to hold the OCR engine
# # This is initialized once at startup and reused for all requests
# # This is important because loading the model is slow (can take several seconds)
# ocr_engine = None
#
# def load_ocr():
#     """
#     Initialize and return a PaddleOCR engine instance.
#
#     PaddleOCR is a deep learning-based OCR toolkit that can detect and recognize
#     text in images. It uses neural networks trained on millions of text samples.
#
#     Parameters:
#     - use_doc_orientation_classify: Automatically detects if the document is rotated
#       (e.g., 90°, 180°, 270°) and corrects it before OCR
#     - use_doc_unwarping: Corrects curved or warped documents (like photos of book pages)
#       Set to False here for performance; enable if needed
#     - use_textline_orientation: Detects if individual text lines are upside down
#     - lang: Language for text recognition ("en" for English)
#
#     Returns:
#     - PaddleOCR instance ready to process images
#     """
#     return PaddleOCR(
#         use_doc_orientation_classify=True,  # Auto-detect document rotation
#         use_doc_unwarping=False,            # Don't correct warped pages (faster)
#         use_textline_orientation=True,      # Detect upside-down text lines
#         lang="en"                           # English language model
#     )
#
# # ================= MIDDLEWARE =================
#
# class TimeoutMiddleware(BaseHTTPMiddleware):
#     """
#     Middleware that adds a timeout to all HTTP requests.
#
#     Middleware in FastAPI/Starlette is code that runs before and after each request.
#     This middleware ensures that no single request takes longer than the specified
#     timeout, preventing the server from hanging indefinitely.
#
#     How it works:
#     1. Wraps the request processing in asyncio.wait_for()
#     2. If processing takes longer than timeout seconds, raises TimeoutError
#     3. Catches the timeout and returns a 408 (Request Timeout) response
#     """
#
#     def __init__(self, app, timeout=60):
#         """
#         Initialize the middleware.
#
#         Args:
#             app: The ASGI application (FastAPI app)
#             timeout: Maximum time in seconds to wait for a request (default: 60)
#         """
#         super().__init__(app)
#         self.timeout = timeout
#
#     async def dispatch(self, request, call_next):
#         """
#         Process each incoming request with a timeout.
#
#         Args:
#             request: The incoming HTTP request
#             call_next: Function that calls the next middleware/route handler
#
#         Returns:
#             Response from the route handler, or 408 if timeout occurs
#         """
#         try:
#             # Wrap the request processing with a timeout
#             # asyncio.wait_for() will raise TimeoutError if call_next takes too long
#             return await asyncio.wait_for(
#                 call_next(request),
#                 timeout=self.timeout
#             )
#         except asyncio.TimeoutError:
#             # If timeout occurs, return a 408 status code
#             return JSONResponse(
#                 status_code=408,  # HTTP 408: Request Timeout
#                 content={
#                     "success": False,
#                     "error": "request_timeout"
#                 }
#             )
#
# # Add the timeout middleware to the application
# # All requests will now have a 60-second timeout
# app.add_middleware(TimeoutMiddleware, timeout=60)
#
# # ================= STARTUP =================
#
# @app.on_event("startup")
# async def startup():
#     """
#     Startup event handler that runs once when the application starts.
#
#     This is crucial for:
#     1. Loading the heavy OCR model into memory (done once, not per request)
#     2. Warming up the model with a dummy prediction (prevents first-request slowness)
#
#     Why warm-up? Deep learning models often have lazy initialization.
#     The first prediction might trigger additional setup, making it slower.
#     By running a dummy prediction at startup, we ensure all subsequent
#     predictions are fast.
#     """
#     global ocr_engine  # Access the global variable
#
#     logger.info("Loading PaddleOCR...")
#
#     # Load the OCR model into memory
#     # This can take 5-30 seconds depending on hardware
#     ocr_engine = load_ocr()
#
#     # Create a dummy black image (200x200 pixels, 3 color channels)
#     # np.zeros creates an array filled with zeros (black pixels)
#     # dtype=np.uint8 means each pixel value is an unsigned 8-bit integer (0-255)
#     # Shape (200, 200, 3) means: 200 rows, 200 columns, 3 channels (BGR)
#     dummy = np.zeros((200, 200, 3), dtype=np.uint8)
#
#     try:
#         # Run a dummy prediction to "warm up" the model
#         # asyncio.to_thread() runs the blocking predict() function in a separate thread
#         # This prevents blocking the async event loop during the warm-up
#         await asyncio.to_thread(
#             ocr_engine.predict,
#             dummy
#         )
#     except Exception:
#         # Ignore any errors during warm-up
#         # The model should still work for real images
#         pass
#
#     logger.info("OCR ready")
#
# # ================= UTILITIES =================
#
# # Initialize python-magic for MIME type detection
# # mime=True tells it to return the MIME type (e.g., "image/jpeg") instead of description
# mime_magic = magic.Magic(mime=True)
#
# def sanitize_filename(name: str) -> str:
#     """
#     Sanitize a filename to prevent directory traversal attacks.
#
#     Security concern: A malicious user could upload a file named "../../../etc/passwd"
#     to try to access system files. os.path.basename() extracts just the filename,
#     removing any directory path components.
#
#     Args:
#         name: Original filename from the upload
#
#     Returns:
#         Safe filename with path components removed
#     """
#     # os.path.basename("/path/to/file.jpg") returns "file.jpg"
#     # If name is None or empty, use "unknown" as fallback
#     return os.path.basename(name or "unknown")
#
# def validate_signature(contents: bytes):
#     """
#     Validate that the uploaded file is actually an allowed image type.
#
#     Why check the signature? Users can rename a .exe file to .jpg, but the file's
#     binary signature (magic bytes) won't change. This function checks the actual
#     file content, not just the extension.
#
#     Args:
#         contents: Raw bytes of the uploaded file
#
#     Raises:
#         ValueError: If the file is not an allowed image type
#     """
#     # Detect the actual MIME type from the file's binary content
#     mime = mime_magic.from_buffer(contents)
#
#     # Check if the detected type is in our allowed list
#     if mime not in ALLOWED_MIMES:
#         raise ValueError("invalid_file_signature")
#
# def decode_image(contents: bytes):
#     """
#     Convert raw image bytes into a NumPy array that OpenCV can process.
#
#     Images are stored as bytes on disk/network, but OpenCV needs them as
#     NumPy arrays for mathematical operations.
#
#     Args:
#         contents: Raw bytes of the image file
#
#     Returns:
#         NumPy array representing the image (shape: height × width × channels)
#
#     Raises:
#         ValueError: If the image cannot be decoded or is too large
#     """
#     # Convert bytes to a NumPy array of unsigned 8-bit integers
#     # This is necessary because cv2.imdecode expects a NumPy array
#     arr = np.frombuffer(contents, np.uint8)
#
#     # Decode the compressed image (JPEG/PNG/WebP) into raw pixel data
#     # cv2.IMREAD_COLOR loads the image with 3 color channels (BGR format)
#     # Note: OpenCV uses BGR order, not RGB!
#     image = cv2.imdecode(
#         arr,
#         cv2.IMREAD_COLOR
#     )
#
#     # If decoding failed (corrupted file, unsupported format), image will be None
#     if image is None:
#         raise ValueError("invalid_image")
#
#     # Get image dimensions
#     # shape[0] = height (number of rows)
#     # shape[1] = width (number of columns)
#     h, w = image.shape[:2]
#
#     # Check if the image is too large (too many pixels)
#     # Large images consume lots of memory and take longer to process
#     if h * w > MAX_PIXELS:
#         raise ValueError(
#             "image_dimensions_too_large"
#         )
#
#     return image
#
# def resize_image(image, max_width=1500):
#     """
#     Resize an image to have a maximum width while maintaining aspect ratio.
#
#     Why resize?
#     - Large images slow down OCR processing
#     - Most text doesn't need ultra-high resolution for recognition
#     - Reduces memory usage
#
#     Args:
#         image: Input image as NumPy array
#         max_width: Maximum allowed width in pixels (default: 1500)
#
#     Returns:
#         Resized image if width > max_width, otherwise original image
#     """
#     h, w = image.shape[:2]
#
#     # If image is already within the width limit, return it unchanged
#     if w <= max_width:
#         return image
#
#     # Calculate the scaling factor to reduce width to max_width
#     # Aspect ratio is preserved: new_height = original_height × scale
#     scale = max_width / w
#
#     # Resize the image using cv2.resize
#     # New dimensions: (new_width, new_height)
#     # interpolation=cv2.INTER_AREA: Best for shrinking images (reduces aliasing)
#     return cv2.resize(
#         image,
#         (max_width, int(h * scale)),  # New width and height
#         interpolation=cv2.INTER_AREA  # Interpolation method for resizing
#     )
#
# def enhance_contrast(image):
#     """
#     Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
#
#     Why enhance contrast?
#     - Poor lighting can make text hard to read
#     - CLAHE improves local contrast without over-amplifying noise
#     - Works well for documents with uneven lighting
#
#     How it works:
#     1. Convert image from BGR to LAB color space
#        - L: Lightness (brightness)
#        - A: Green-Red color component
#        - B: Blue-Yellow color component
#     2. Apply CLAHE only to the L channel (lightness)
#        - This enhances contrast without affecting colors
#     3. Convert back to BGR
#
#     Args:
#         image: Input image in BGR format
#
#     Returns:
#         Image with enhanced contrast
#     """
#     # Convert from BGR to LAB color space
#     # LAB separates lightness from color information
#     lab = cv2.cvtColor(
#         image,
#         cv2.COLOR_BGR2LAB
#     )
#
#     # Split LAB into three separate channels
#     l, a, b = cv2.split(lab)
#
#     # Create a CLAHE object
#     # clipLimit=2.0: Limits contrast enhancement to prevent noise amplification
#     # tileGridSize=(8, 8): Divides image into 8×8 tiles for local histogram equalization
#     clahe = cv2.createCLAHE(
#         clipLimit=2.0,
#         tileGridSize=(8, 8)
#     )
#
#     # Apply CLAHE to the lightness channel
#     l = clahe.apply(l)
#
#     # Merge the enhanced L channel with original A and B channels
#     # Convert back to BGR color space
#     return cv2.cvtColor(
#         cv2.merge([l, a, b]),
#         cv2.COLOR_LAB2BGR
#     )
#
# def sharpen(image):
#     """
#     Sharpen an image using unsharp masking.
#
#     Why sharpen?
#     - Blurry images (from camera shake, poor focus) are harder for OCR
#     - Sharpening enhances edges, making text boundaries clearer
#
#     How unsharp masking works:
#     1. Create a blurred version of the image
#     2. Subtract the blur from the original (enhances edges)
#     3. Combine with weighted addition
#
#     Formula: sharpened = original × 1.5 - blur × 0.5
#
#     Args:
#         image: Input image
#
#     Returns:
#         Sharpened image
#     """
#     # Create a blurred version using Gaussian blur
#     # (0, 0) means kernel size is calculated from sigma
#     # sigma=2 controls the amount of blur
#     blur = cv2.GaussianBlur(
#         image,
#         (0, 0),  # Kernel size (auto-calculated from sigma)
#         2        # Standard deviation (sigma) for Gaussian kernel
#     )
#
#     # Combine original and blurred images using weighted addition
#     # dst = src1 × alpha + src2 × beta + gamma
#     # Here: dst = image × 1.5 + blur × (-0.5) + 0
#     # Negative weight on blur effectively subtracts it, enhancing edges
#     return cv2.addWeighted(
#         image,   # First image
#         1.5,     # Weight for first image
#         blur,    # Second image
#         -0.5,    # Weight for second image (negative = subtraction)
#         0        # Scalar added to each sum
#     )
#
# def preprocess(contents):
#     """
#     Complete preprocessing pipeline for an uploaded image.
#
#     Steps:
#     1. Validate file type (security)
#     2. Decode image bytes to NumPy array
#     3. Record original dimensions (for coordinate mapping later)
#     4. Resize if too large
#
#     Args:
#         contents: Raw bytes of the uploaded image file
#
#     Returns:
#         Tuple of (processed_image, original_width, original_height)
#
#     Raises:
#         ValueError: If validation fails or image is invalid
#     """
#     # Step 1: Check if file is actually an allowed image type
#     validate_signature(contents)
#
#     # Step 2: Convert bytes to OpenCV-compatible NumPy array
#     image = decode_image(contents)
#
#     # Store original dimensions before any modifications
#     # These are needed later to map OCR coordinates back to the original image
#     original_h, original_w = image.shape[:2]
#
#     # Step 3: Resize if necessary (maintains aspect ratio)
#     image = resize_image(image)
#
#     # Return processed image along with original dimensions
#     return image, original_w, original_h
#
# # ================= OCR =================
#
# def run_ocr_multi_pass(image):
#     """
#     Run OCR multiple times with different image enhancements.
#
#     Why multi-pass?
#     - Different preprocessing works better for different images
#     - Combining results increases accuracy
#     - If one method fails, others might succeed
#
#     Three variants are tried:
#     1. Original image (baseline)
#     2. Contrast-enhanced version (better for low-light/poor contrast)
#     3. Sharpened version (better for blurry images)
#
#     Args:
#         image: Preprocessed image as NumPy array
#
#     Returns:
#         List of OCR results from each variant
#     """
#     # Create three variants of the image
#     variants = [
#         image,                        # Variant 1: Original
#         enhance_contrast(image.copy()),  # Variant 2: Enhanced contrast
#         sharpen(image.copy())            # Variant 3: Sharpened
#     ]
#
#     # Store results from all variants
#     all_results = []
#
#     # Process each variant
#     for img in variants:
#         try:
#             # Run OCR on this variant
#             # ocr_engine.predict() returns detection and recognition results
#             result = ocr_engine.predict(img)
#             all_results.append(result)
#         except Exception as exc:
#             # Log warning if a variant fails, but continue with others
#             logger.warning(exc)
#
#     return all_results
#
# def process_results(results, width, height):
#     """
#     Process and combine OCR results from multiple passes.
#
#     This function:
#     1. Extracts text, confidence scores, and bounding boxes from raw OCR output
#     2. Filters out low-confidence detections
#     3. Sorts detections by position (top-to-bottom, left-to-right reading order)
#     4. Combines all detected text into a single string
#     5. Calculates average confidence score
#
#     Args:
#         results: List of OCR results from run_ocr_multi_pass()
#         width: Original image width (for metadata)
#         height: Original image height (for metadata)
#
#     Returns:
#         Dictionary containing:
#         - image_width, image_height: Original dimensions
#         - full_text: All detected text concatenated
#         - average_confidence: Mean confidence across all detections
#         - detections: List of individual text detections with positions
#     """
#     # List to store all valid text detections
#     detections = []
#
#     # Iterate through results from each variant
#     for batch in results:
#         # Skip empty results
#         if not batch:
#             continue
#
#         # Each batch may contain multiple pages (for multi-page documents)
#         for page in batch:
#             try:
#                 # Extract recognition results from the page
#                 # rec_texts: List of recognized text strings
#                 # rec_scores: Confidence scores for each text (0.0 to 1.0)
#                 # dt_polys: Polygon coordinates for each detected text region
#                 texts = page.get("rec_texts", [])
#                 scores = page.get("rec_scores", [])
#                 polys = page.get("dt_polys", [])
#
#                 # Process each detected text region
#                 for text, score, poly in zip(texts, scores, polys):
#                     # Filter out low-confidence detections
#                     # Threshold of 0.5 means we only keep detections with ≥50% confidence
#                     if score < 0.5:
#                         continue
#
#                     # Extract x and y coordinates from the polygon
#                     # poly is a list of [x, y] points forming a quadrilateral around the text
#                     xs = [p[0] for p in poly]  # All x-coordinates
#                     ys = [p[1] for p in poly]  # All y-coordinates
#
#                     # Create a detection dictionary with text and position info
#                     detections.append({
#                         "text": text.strip(),  # Remove leading/trailing whitespace
#                         "confidence": round(float(score), 3),  # Round to 3 decimal places
#                         "x": int(min(xs)),  # Leftmost x-coordinate (bounding box)
#                         "y": int(min(ys)),  # Topmost y-coordinate (bounding box)
#                         "width": int(max(xs) - min(xs)),  # Width of bounding box
#                         "height": int(max(ys) - min(ys)),  # Height of bounding box
#                         # Full polygon coordinates (useful for highlighting text in UI)
#                         "bounding_box": poly.tolist() if hasattr(poly, "tolist") else poly
#                     })
#
#             except Exception:
#                 # Skip this page if there's an error parsing it
#                 continue
#
#     # Sort detections by position for natural reading order
#     # Primary sort: by y-coordinate (top to bottom)
#     # Secondary sort: by x-coordinate (left to right) for same line
#     detections.sort(
#         key=lambda x: (
#             x["y"],  # Sort by vertical position first
#             x["x"]   # Then by horizontal position
#         )
#     )
#
#     # Combine all detected text into a single string
#     # Join with spaces to separate words/lines
#     full_text = " ".join(
#         d["text"]
#         for d in detections
#         if d["text"]  # Only include non-empty text
#     )
#
#     # Calculate average confidence across all detections
#     avg = 0.0
#     if detections:
#         avg = round(
#             sum(d["confidence"] for d in detections) / len(detections),
#             3
#         )
#
#     # Return structured result
#     return {
#         "image_width": width,
#         "image_height": height,
#         "full_text": full_text,
#         "average_confidence": avg,
#         "detections": detections
#     }
#
# # ================= ENDPOINTS =================
#
# @app.get("/health")
# async def health():
#     """
#     Health check endpoint.
#
#     Used by load balancers and monitoring systems to verify the service is running.
#     Always returns 200 OK if the server is up, regardless of model state.
#
#     Returns:
#         JSON with status field
#     """
#     return {"status": "healthy"}
#
# @app.get("/ready")
# async def ready():
#     """
#     Readiness check endpoint.
#
#     Used to verify the service is fully initialized and ready to process requests.
#     Checks if the OCR model has been loaded successfully.
#
#     Returns:
#         JSON with ready boolean indicating if model is loaded
#     """
#     return {
#         "ready": ocr_engine is not None
#     }
#
# @app.post("/ocr")
# async def ocr_endpoint(
#     request: Request,
#     files: List[UploadFile] = File(...)
# ):
#     """
#     Main OCR endpoint that processes uploaded images.
#
#     Accepts multiple image files, runs OCR on each, and returns structured results.
#
#     Args:
#         request: FastAPI request object (for logging/metadata)
#         files: List of uploaded image files
#
#     Returns:
#         JSON response with:
#         - success: Overall success status
#         - processing_time: Total time taken in seconds
#         - results: List of results for each image (success/failure + OCR data)
#     """
#     # Validate number of files
#     if len(files) > MAX_FILES:
#         return JSONResponse(
#             status_code=400,  # Bad Request
#             content={
#                 "success": False,
#                 "error": "too_many_files"
#             }
#         )
#
#     # Generate unique ID for this request (for logging/tracking)
#     request_id = str(uuid.uuid4())
#
#     # List to accumulate results for all files
#     results = []
#
#     # Track total bytes uploaded across all files
#     total_bytes = 0
#
#     # Record start time for performance measurement
#     start = time.time()
#
#     # Process each uploaded file
#     for file in files:
#         # Sanitize filename for security
#         safe_name = sanitize_filename(file.filename)
#
#         try:
#             # Read the entire file content into memory
#             # Note: For very large files, consider streaming instead
#             contents = await file.read()
#
#             # Accumulate total bytes
#             total_bytes += len(contents)
#
#             # Check if total upload size exceeds limit
#             if total_bytes > MAX_TOTAL_UPLOAD_MB * 1024 * 1024:
#                 raise ValueError("total_upload_limit_exceeded")
#
#             # Check if individual file size exceeds limit
#             if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
#                 raise ValueError("file_too_large")
#
#             # Step 1: Preprocess the image (validate, decode, resize)
#             # asyncio.to_thread() runs CPU-bound code in a thread pool
#             # This prevents blocking the async event loop
#             image, w, h = await asyncio.to_thread(
#                 preprocess,
#                 contents
#             )
#
#             # Step 2: Run OCR with multiple preprocessing variants
#             raw = await asyncio.to_thread(
#                 run_ocr_multi_pass,
#                 image
#             )
#
#             # Step 3: Process and structure the OCR results
#             processed = await asyncio.to_thread(
#                 process_results,
#                 raw,
#                 w,
#                 h
#             )
#
#             # Add successful result to the list
#             results.append({
#                 "image_name": safe_name,
#                 "success": True,
#                 **processed  # Unpack the processed dict into this dict
#             })
#
#         except Exception as exc:
#             # Log the error with request ID and filename for debugging
#             logger.exception(
#                 "%s failed %s",
#                 request_id,
#                 safe_name
#             )
#
#             # Add failure result to the list
#             results.append({
#                 "image_name": safe_name,
#                 "success": False,
#                 "error": str(exc)
#             })
#
#     # Calculate total processing time
#     processing_time = round(time.time() - start, 2)
#
#     # Return final response
#     return {
#         "success": True,
#         "processing_time": processing_time,
#         "results": results
#     }


"""
Production FastAPI + PaddleOCR 2.8.1 (PP-OCRv4) OCR Microservice
Stable Linux/Kubuntu implementation.
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
    """
    Initialize PaddleOCR 2.8.1.
    PP-OCRv4 is the default, highly stable architecture in this version.
    """
    return PaddleOCR(
        use_angle_cls=True,  # Stable textline orientation classifier in 2.x
        lang="en",  # English recognition
        show_log=False,  # Suppress Paddle's verbose internal C++ logs
        use_gpu=False  # Force CPU inference
    )


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
        # 2.x API uses .ocr() instead of .predict()
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
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def sharpen(image):
    blur = cv2.GaussianBlur(image, (0, 0), 2)
    return cv2.addWeighted(image, 1.5, blur, -0.5, 0)


def preprocess(contents):
    validate_signature(contents)
    image = decode_image(contents)
    original_h, original_w = image.shape[:2]
    image = resize_image(image)
    return image, original_w, original_h


# ================= OCR =================
def run_ocr_multi_pass(image):
    variants = [
        ("original", image),
        ("contrast", enhance_contrast(image.copy())),
        ("sharpen", sharpen(image.copy()))
    ]
    all_results = []
    for pass_name, img in variants:
        start_pass = time.time()
        try:
            # PaddleOCR 2.x API
            result = ocr_engine.ocr(img, cls=True)
            duration_pass = time.time() - start_pass
            logger.info("OCR pass completed",
                        extra={"pass_variant": pass_name, "duration_ms": round(duration_pass * 1000, 2)})
            all_results.append(result)
        except Exception as exc:
            duration_pass = time.time() - start_pass
            logger.warning("OCR pass failed",
                           extra={"pass_variant": pass_name, "duration_ms": round(duration_pass * 1000, 2),
                                  "error": str(exc)})
    return all_results


def process_results(results, width, height):
    detections = []
    for batch in results:
        # In 2.x, batch is a list containing one element (the page results)
        if not batch or not batch[0]:
            continue

        for line in batch[0]:
            try:
                # 2.x format: line = [ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence) ]
                poly = line[0]
                text = line[1][0]
                score = line[1][1]

                if score < 0.5:
                    continue

                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]

                detections.append({
                    "text": text.strip(),
                    "confidence": round(float(score), 3),
                    "x": int(min(xs)),
                    "y": int(min(ys)),
                    "width": int(max(xs) - min(xs)),
                    "height": int(max(ys) - min(ys)),
                    "bounding_box": poly
                })
            except Exception:
                continue

    detections.sort(key=lambda x: (x["y"], x["x"]))
    full_text = " ".join(d["text"] for d in detections if d["text"])

    avg = 0.0
    if detections:
        avg = round(sum(d["confidence"] for d in detections) / len(detections), 3)

    return {
        "image_width": width,
        "image_height": height,
        "full_text": full_text,
        "average_confidence": avg,
        "detections": detections
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

    results = []
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
            raw = await asyncio.to_thread(run_ocr_multi_pass, image)
            duration_ocr = time.time() - start_ocr
            logger.info("OCR completed",
                        extra={"image_filename": safe_name, "duration_ms": round(duration_ocr * 1000, 2)})

            start_post = time.time()
            processed = await asyncio.to_thread(process_results, raw, w, h)
            duration_post = time.time() - start_post
            logger.info("Postprocess completed",
                        extra={"image_filename": safe_name, "duration_ms": round(duration_post * 1000, 2),
                               "detections_count": len(processed.get("detections", []))})

            results.append({"image_name": safe_name, "success": True, **processed})

        except Exception as exc:
            logger.exception("File processing failed", extra={"image_filename": safe_name})
            results.append({"image_name": safe_name, "success": False, "error": str(exc)})

    processing_time = round(time.time() - start, 2)
    success_count = sum(1 for r in results if r.get("success"))
    fail_count = len(results) - success_count

    logger.info("Request summary",
                extra={"total_files": len(files), "success_count": success_count, "fail_count": fail_count,
                       "total_duration_s": processing_time, "total_bytes_uploaded": total_bytes})

    return {"success": True, "processing_time": processing_time, "results": results}