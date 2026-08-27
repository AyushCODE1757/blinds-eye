# Spatial Navigation System - Team Integration Guide

The ML Backend Engine is complete and running locally on `http://127.0.0.1:8000`. Follow the instructions below to hook up the frontend interface, spatial audio, speech alerts, and haptic feedback.

---

## 1. Backend Server Endpoints

Ensure the ML backend server is active (`python server.py`) before testing.

* **Live Video Stream (RGB + Depth Map):** `http://127.0.0.1:8000/video_feed`
* **Detection Data API:** `http://127.0.0.1:8000/api/detect`

### JSON Data Format (`/api/detect`)
```json
{
  "status": "ALERT",          // "ALERT" or "CLEAR"
  "alert_side": "left",       // "left", "right", "center", or "none"
  "obstacles": [
    {
      "label": "chair",       // Detected class
      "proximity": 640,       // Relative depth value (higher = closer)
      "direction": "left",    // "left", "center", or "right"
      "bbox": [50, 100, 200, 350]
    }
  ]
}