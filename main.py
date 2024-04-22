import os
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from pydantic import BaseModel


class LogoDetector:
    def __init__(self, model_path="runs/detect/yolov8n_custom/weights/best.pt"):
        self.model = YOLO(model_path)

    def detect_logos(self, input_data):
        if isinstance(input_data, str):  # If input is a file path
            print(input_data)
            response = requests.get(input_data)
            image_bytes = BytesIO(response.content)
            image = cv2.imdecode(
                np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR
            )
        elif isinstance(input_data, Image.Image):  # If input is a PIL image
            image = np.array(input_data)
        else:
            raise ValueError("Input data must be a file path or a PIL image.")

        results = self.model.predict(image)

        logo_list = []
        names = self.model.names
        for r in results:
            for c, conf, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh):
                logo_name = names[int(c)]
                confidence = float(conf)
                print(box)
                logo_list.append(
                    {
                        "logo_name": logo_name,
                        "confidence": confidence,
                        "box": box.tolist(),
                    }
                )
        return logo_list


# FastAPI Application


app = FastAPI()


class URLSchema(BaseModel):
    url: str


@app.post("/detect_logos_file/")
async def detect_logos_file(file: UploadFile):
    logo_detector = LogoDetector()
    image = Image.open(BytesIO(await file.read()))
    results = logo_detector.detect_logos(image)
    return results


@app.post("/detect_logos_url/")
async def detect_logos_url(req: URLSchema):
    logo_detector = LogoDetector()
    url = req.url
    print(url)
    results = logo_detector.detect_logos(url)
    return results
