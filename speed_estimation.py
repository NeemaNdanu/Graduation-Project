from ultralytics import YOLO
import cv2
import streamlit as st
import os
import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_speed_estimation(video_path, output_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    tracker = DeepSort(max_age=30)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()

    prev_positions = {}
    track_class_map = {}  # Maps track_id to class_name
    pixel_to_meter = 0.05  # Approximate conversion

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            class_name = model.names[int(cls)]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

        tracks = tracker.update_tracks(detections, frame=frame)

        for tr in tracks:
            if not tr.is_confirmed():
                continue

            x, y, w, h = map(int, tr.to_ltwh())
            cx, cy = x + w // 2, y + h // 2
            tid = tr.track_id

            # Get the class name associated with this track
            class_name = tr.det_class if hasattr(tr, 'det_class') else 'object'
            if isinstance(class_name, int):  # If it returns an index
                class_name = model.names[class_name]
            elif not isinstance(class_name, str):
                class_name = 'object'

            track_class_map[tid] = class_name

            if tid in prev_positions:
                px, py = prev_positions[tid]
                dist = np.hypot(cx - px, cy - py)
                speed = dist * pixel_to_meter * fps * 3.6  # km/h
            else:
                speed = 0

            prev_positions[tid] = (cx, cy)
            label = f"{track_class_map[tid]} | {int(speed)} km/h"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    out.release()
    st.success("✅ Speed estimation completed.")
    st.video(output_path)
