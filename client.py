import cv2
import requests
import time

# Replace localhost with your Azure VM IP (e.g., "http://52.237.88.144:3000/predict")
SERVER_URL = "http://52.237.88.144:3000/predict"

def start_camera_feed():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Streaming to {SERVER_URL}...")
    print("Press 'q' in the video window to stop.\n")

    # Track time to avoid spamming the server
    last_sent_time = time.time()
    send_interval = 2.5  # seconds

    try:
        while True:
            # 1. Constantly grab frames to keep the video smooth
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # 2. Show the live feed in a local window
            cv2.imshow("Live Camera Feed", frame)

            # 3. Check if 2.5 seconds have passed since the last request
            current_time = time.time()
            if current_time - last_sent_time >= send_interval:
                last_sent_time = current_time  # Reset the timer
                
                # Compress the frame
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                    
                    try:
                        # 4. Send the POST request (with a timeout so the video doesn't freeze)
                        response = requests.post(SERVER_URL, files=files, timeout=2.0)
                        
                        if response.status_code == 200:
                            data = response.json()
                            timestamp = time.strftime('%H:%M:%S')
                            
                            if data['faces_detected'] == 0:
                                print(f"[{timestamp}] No faces detected.")
                            else:
                                print(f"[{timestamp}] Detected {data['faces_detected']} face(s):")
                                for face in data.get('results', []):
                                    print(f"  -> Name: {face['name']} | Confidence: {face['confidence']:.4f}")
                        else:
                            print(f"Server error: {response.status_code}")

                    except requests.exceptions.RequestException as e:
                        print(f"[{time.strftime('%H:%M:%S')}] Connection failed. Is the server running?")

            # 5. Required by OpenCV to actually update the video window
            # Also listens for the 'q' key to quit gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q' pressed. Stopping camera feed...")
                break

    except KeyboardInterrupt:
        print("\nStopping camera feed...")
    
    finally:
        # Clean up resources and close the window
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_feed()