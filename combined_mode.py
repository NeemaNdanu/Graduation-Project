from ultralytics import YOLO
import cv2
import streamlit as st
import os
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_combined_mode(video_path, output_path, model_path):
    st.info("🔁 Running combined object detection and speed estimation...")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    tracker = DeepSort(max_age=30)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    object_speeds = {}
    last_positions = {}
    last_time = time.time()
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "vehicle"))

        tracks = tracker.update_tracks(detections, frame=frame)
        current_time = time.time()
        time_diff = current_time - last_time
        last_time = current_time

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cx, cy = int((l + r) / 2), int((t + b) / 2)

            if track_id in last_positions:
                px, py = last_positions[track_id]
                pixel_dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                speed = (pixel_dist / time_diff) * 0.05
                object_speeds[track_id] = speed
            last_positions[track_id] = (cx, cy)

            speed_text = f"ID {track_id} | Speed: {object_speeds.get(track_id, 0):.1f} km/h"
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, speed_text, (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    out.release()
    st.success("✅ Combined Model Completed")
    return output_path