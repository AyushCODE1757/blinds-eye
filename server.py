import os
# 1. Prevent PyTorch & OpenCV multi-threading deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import threading
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from ml_engine import NavigationMLEngine

# Disable extra OpenCV internal threads to prevent GIL locks
cv2.setNumThreads(0)

app = FastAPI(title="Spatial Navigation ML Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global State Variables
engine = None
cap = None
latest_payload = {"status": "CLEAR", "alert_side": "none", "obstacles": []}
latest_jpeg_bytes = None
lock = threading.Lock()
is_running = True

def camera_processing_loop():
    """Background thread that continuously reads frames and runs ML processing."""
    global latest_payload, latest_jpeg_bytes, is_running

    print("Camera processing thread started...")
    while is_running and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        try:
            # Run ML Engine
            payload, annotated_frame, depth_map = engine.process_frame(frame)

            # Build side-by-side view (RGB + Depth map)
            combined = np.hstack((annotated_frame, depth_map))
            _, buffer = cv2.imencode('.jpg', combined)
            jpeg_bytes = buffer.tobytes()

            with lock:
                latest_payload = payload
                latest_jpeg_bytes = jpeg_bytes

        except Exception as e:
            print(f"Error in processing loop: {e}")

        time.sleep(0.02)  # Limit CPU load (~30 FPS)

@app.on_event("startup")
def startup_event():
    """Runs automatically when FastAPI starts."""
    global engine, cap
    print("Loading ML Engine models...")
    engine = NavigationMLEngine()

    print("Opening system webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("WARNING: Could not open webcam index 0!")
    else:
        print("Webcam successfully opened.")

    # Start background processing thread
    thread = threading.Thread(target=camera_processing_loop, daemon=True)
    thread.start()
    print("Server ready! Access http://127.0.0.1:8000/api/detect")

@app.on_event("shutdown")
def shutdown_event():
    global is_running, cap
    is_running = False
    if cap and cap.isOpened():
        cap.release()

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/")
def root():
    return {"status": "ML Server Running", "endpoints": ["/api/detect", "/video_feed"]}

@app.get("/api/detect")
def get_detection_data():
    with lock:
        return latest_payload

def mjpeg_generator():
    while True:
        with lock:
            if latest_jpeg_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_jpeg_bytes + b'\r\n')
        time.sleep(0.04)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)