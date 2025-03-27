import gradio as gr
import cv2
import numpy as np

import torch
from gradio_webrtc import WebRTC
from twilio.rest import Client
import asyncio

from ultralytics import YOLO

import os
import time

# Ensure cache directories exist
os.environ["GRADIO_TEMP_DIR"] = "./gradio_cache"
os.environ["GRADIO_CACHE_DIR"] = "./gradio_cache"
os.makedirs("./gradio_cache", exist_ok=True)

# Setting up the environment to use only GPU 2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Additional GPU settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

model = YOLO("models/best.pt").to(device)
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

if account_sid and auth_token:
    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    rtc_configuration = {
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    }

def process_frame_webcam(input_image, conf_threshold=0.3):
    input_image = cv2.flip(input_image, 1)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    results = model(input_image, conf=conf_threshold)
    result = results[0] if isinstance(results, list) else results
    segmented_frame = result.plot()
    return segmented_frame

def gradio_infer(input_image, conf_threshold):
    try:
        start = time.time()
        segmented_frame = process_frame_webcam(input_image, conf_threshold)
        segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)
        fps = 1 / (time.time() - start)
        height, width, _ = segmented_frame.shape
        cv2.putText(segmented_frame, f"FPS: {fps:.2f}",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        return segmented_frame #cv2.resize(segmented_frame, (500, 500))
    except Exception as e:
        print(f"Error during processing: {e}")
        height, width, _ = input_image.shape
        return np.zeros((height, width, 3), dtype=np.uint8)

css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv11 Webcam Stream
    </h1>
    """)
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream", rtc_configuration=rtc_configuration)
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )

        image.stream(
            fn=gradio_infer, inputs=[image, conf_threshold], outputs=[image], time_limit=1000
        )

if __name__ == "__main__":
    demo.launch(share=True)