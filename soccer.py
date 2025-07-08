import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

# Load video and model
video = cv2.VideoCapture("15sec_input_720p.mp4")
model = YOLO("best.pt")  # Replace with your actual model path

# Player ID management
player_db = {}
next_id = 0
DIST_THRESHOLD = 50  # pixels

def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def match_player(center):
    global next_id
    for pid, prev_center in player_db.items():
        if euclidean_distance(center, prev_center) < DIST_THRESHOLD:
            player_db[pid] = center
            return pid
    player_db[next_id] = center
    next_id += 1
    return next_id - 1

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    for result in results:
        boxes = result.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        bboxes = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for cls_id, bbox, conf in zip(class_ids, bboxes, confs):
            label = model.names[cls_id]
            if label.lower() == "player":
                x1, y1, x2, y2 = map(int, bbox)
                center = get_center(x1, y1, x2, y2)
                player_id = match_player(center)

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Player {player_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Player Re-ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
