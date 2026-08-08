import streamlit as st
import requests

st.set_page_config(
    page_title="Vehicle Recognition",
    page_icon="",
    layout="wide"
)

st.title("Vehicle Recognition System")
st.markdown("Upload a vehicle image to detect the vehicle type and read the license plate.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a vehicle"
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width='stretch')
    
    if st.button("Analyze", type="primary"):
        with st.spinner("Processing image..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(
                    "http://localhost:8000/predict",
                    files=files
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    with col2:
                        st.subheader("Detection Results")
                        st.write(f"**Total Vehicles Found:** {data['total']}")
                        
                        if data['total'] == 0:
                            st.warning("No vehicles detected in this image.")
                        else:
                            for i, vehicle in enumerate(data['vehicles'], 1):
                                st.divider()
                                st.write(f"**Vehicle {i}:** {vehicle['type']}")
                                st.write(f"   **Confidence:** {vehicle['confidence']:.2%}")
                                st.write(f"   **License Plate:** {vehicle['plate'] or 'Not detected'}")
                else:
                    st.error(f"Server error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Make sure the server is running.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.info("Upload an image to get started.")

st.divider()
st.caption("Built with YOLO + EasyOCR + FastAPI | Powered by Streamlit")