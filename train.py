from ultralytics import YOLO
import os
import shutil


def train():
    model = YOLO("yolov8n.pt")
    directory_to_delete = "runs"
    if os.path.exists(directory_to_delete) and os.path.isdir(directory_to_delete):
        try:
            shutil.rmtree(directory_to_delete)  # Removes the directory and its contents
            print(f"Directory '{directory_to_delete}' has been deleted.")
        except OSError as e:
            print(f"Error: {e}")
    else:
        print(f"Directory '{directory_to_delete}' does not exist.")

    # Training.
    results = model.train(
        data="data.yaml", imgsz=640, epochs=100, batch=8, name="yolov8n_custom"
    )
