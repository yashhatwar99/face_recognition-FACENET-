import streamlit as st
import requests

# Point to the FastAPI server
SERVER_URL = "http://localhost:3000/predict"

st.title("Face Recognition Client")
st.write("Take a photo to send to the server for prediction.")

camera_image = st.camera_input("Camera Feed")


if camera_image is not None:
    files = {'file': ('frame.jpg', camera_image.getvalue(), 'image/jpeg')}
    try:
        response = requests.post(SERVER_URL, files=files)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', []) 
            if len(results) == 0:
                st.warning("No faces detected.")
            else:
                st.success(f"Detected {len(results)} face(s):")
                for face in results:
                    st.write(f"**Name:** {face['name']} | **Confidence:** {face['confidence']:.4f}")
        else:
            st.error(f"Server error: {response.status_code}")
    except requests.exceptions.RequestException:
        st.error("Connection failed. Is the server running?")

        
