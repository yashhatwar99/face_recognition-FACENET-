import streamlit as st
import requests
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

SERVER_URL = "http://localhost:3000/predict"

st.set_page_config(page_title="Live Face Recognition", layout="wide")
st.title("Real-Time Face Recognition")
st.write("Streaming live video to the FastAPI backend.")

class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_results = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the WebRTC frame to an OpenCV image
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Process every 5th frame to prevent server overload/lag
        if self.frame_count % 5 == 0:
            # Compress the frame before sending to API
            success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                try:
                    # 0.5s timeout: if API is busy, drop the API request and keep video smooth
                    response = requests.post(SERVER_URL, files=files, timeout=0.5)
                    if response.status_code == 200:
                        data = response.json()
                        self.last_results = data.get('results', [])
                except requests.exceptions.RequestException:
                    pass # Keep using old bounding boxes if API lags

        # Draw the most recent results on EVERY frame so the video looks smooth
        for face in self.last_results:
            # Ensure the backend actually returned a bounding box
            if 'bounding_box' in face:
                x1, y1, x2, y2 = face['bounding_box']
                name = face['name']
                conf = face['confidence']
                
                # Draw Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw Label
                label = f"{name} ({conf:.2f})"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Return the modified frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the continuous WebRTC streamer
webrtc_streamer(
    key="face-streamer",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)