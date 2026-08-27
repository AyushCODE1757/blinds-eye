import cv2
import numpy as np
import torch
from ultralytics import YOLO

class NavigationMLEngine:
    def __init__(self):
        print("Initializing ML Engine: Loading YOLOv8n and MiDaS Small...")
        
        # 1. Load Object Detection Model
        self.yolo = YOLO("yolov8n.pt")

        # 2. Load Monocular Depth Estimation Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device).eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def process_frame(self, frame):
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # A. Run Monocular Depth Estimation
        input_batch = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        # B. Run YOLOv8 Object Detection
        results = self.yolo(frame, verbose=False)[0]
        detected_obstacles = []
        alert_triggered = False
        alert_side = "center"

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.yolo.names[int(box.cls[0])]

            # Slice depth map within object bounding box
            crop = depth_map[y1:y2, x1:x2]
            if crop.size > 0:
                proximity = int(np.median(crop))
                
                center_x = (x1 + x2) / 2
                direction = "left" if center_x < w / 3 else ("right" if center_x > 2 * w / 3 else "center")

                # ==================== ADD THESE 2 LINES ====================
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({proximity})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # ===========================================================

                detected_obstacles.append({
                    "label": label,
                    "proximity": proximity,
                    "direction": direction,
                    "bbox": [x1, y1, x2, y2]
                })

                if proximity > 600:
                    alert_triggered = True
                    alert_side = direction

        # Prepare normalized depth map for UI visualization
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

        payload = {
            "status": "ALERT" if alert_triggered else "CLEAR",
            "alert_side": alert_side if alert_triggered else "none",
            "obstacles": detected_obstacles
        }

        return payload, frame, depth_colormap
# Quick standalone local test loop
if __name__ == "__main__":
    engine = NavigationMLEngine()
    cap = cv2.VideoCapture(0)

    print("Starting ML test feed... Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        payload, annotated_frame, depth_map = engine.process_frame(frame)
        
        # Display RGB and Depth side-by-side
        combined = np.hstack((annotated_frame, depth_map))
        cv2.imshow("ML Engine Test - Press Q to exit", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()