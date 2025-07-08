import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import csv

# Load video and model
video_path = "15sec_input_720p.mp4"
video = cv2.VideoCapture(video_path)
model = YOLO("best.pt")

# Video saving setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_tracked_video.mp4', fourcc, fps, (frame_width, frame_height))

# CSV setup
csv_file = open('player_tracking_data.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'ID', 'Label', 'x1', 'y1', 'x2', 'y2'])  # Header

# Player ID tracking
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

frame_number = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame_number += 1

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

                # Write to CSV
                csv_writer.writerow([frame_number, player_id, label, x1, y1, x2, y2])

    # Write to video
    out.write(frame)

    # Show preview (optional)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
