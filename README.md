# soccer_player_tarcking

This project is focused on tracking soccer players using a YOLOv11-based object detection model. It takes a 15-second match video and applies object detection to identify and assign consistent IDs to players across frames, even when they leave and re-enter the camera's view. The system generates two outputs: one is a video file showing bounding boxes and player IDs in real-time, and the other is a CSV file logging each player’s frame-wise position with ID, label, and bounding box coordinates.

To run this project, you’ll need to install some Python dependencies. These include ultralytics for YOLOv11, opencv-python for video processing, numpy and scipy for mathematical operations, and matplotlib if you plan to visualize further. Optionally, if you wish to integrate more advanced tracking, you can install deep_sort_realtime, which adds re-identification support for players even in crowded or overlapping frames.

Once dependencies are installed, make sure you have the trained YOLOv11 model saved as best.pt, and place your soccer video file named 15sec_input_720p.mp4 in the same directory as the Python script. Save the code in a script file, for example, soccer_tracking.py. Then simply run it using python soccer_tracking.py.

The script will process the video frame by frame, detect players using YOLOv11, and assign consistent IDs to them using a basic centroid distance approach. Each frame is written to a new video file, output_tracked_video.mp4, with bounding boxes and ID labels overlaid. Simultaneously, for every player detected in each frame, the system records the frame number, player ID, label, and bounding box coordinates into a CSV file named player_tracking_data.csv. A sample entry in this file would look like: Frame: 1, ID: 0, Label: player, x1: 123, y1: 58, x2: 178, y2: 210.

This project demonstrates a basic re-identification method that can be extended in various ways. You may choose to track other entities such as the referee or the ball by modifying the detection filter. If greater accuracy and stability are needed, integrating Deep SORT would offer improved tracking performance. You can also take this data further by generating heatmaps or visualizing player movements across the pitch over time.

In summary, this project showcases how modern object detection combined with simple tracking logic can be used effectively to analyze sports footage, providing both visual insights and structured data for deeper analysis.

