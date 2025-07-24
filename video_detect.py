from ultralytics import YOLO
import cv2
import streamlit as st
import tempfile
import os
import numpy as np

def run_object_detection(video_path, output_path, model_path):
    st.info("🚗 Running object detection...")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            class_id = int(cls)
            class_name = model.names[class_id]

            # Assign color: red for accident, blue otherwise
            color = (0, 0, 255) if class_name.lower() == "accident" else (255, 0, 0)

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw filled background for text
            text = f"{class_name} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (int(x1), int(y1)-text_height-10), (int(x1) + text_width, int(y1)), color, -1)
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Draw text
            cv2.putText(frame, text, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    out.release()

    st.success("✅ Object detection completed.")
    st.video(output_path)
