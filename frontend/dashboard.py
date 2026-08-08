import streamlit as st
import requests
import streamlit.components.v1 as components

# --- CONFIG ---
# This is the only place you need to change the backend address.
# Update this if your server's IP or port changes.
API_URL = "https://192.168.1.8:8000"
PREDICT_URL = f"{API_URL}/predict"
# --- END CONFIG ---

# --- Page Setup ---
st.set_page_config(
    page_title="Vehicle Recognition System",
    page_icon="",
    layout="wide"
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a mode:",
    ["Upload Photo", "Live Camera"]
)

# --- Page 1: Upload Photo ---
if page == "Upload Photo":
    st.title("Upload a Vehicle Photo")
    st.markdown("Upload an image to detect vehicles and read license plates.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width='stretch')

        if st.button("Analyze", type="primary"):
            with st.spinner("Processing..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    # verify=False skips SSL cert validation — fine for a self-signed
                    # cert on your own LAN, but do not use this if the server is ever
                    # exposed beyond your local network.
                    response = requests.post(
                        PREDICT_URL,
                        files=files,
                        verify=False
                    )

                    if response.status_code == 200:
                        data = response.json()

                        with col2:
                            st.subheader("Detection Results")
                            st.write(f"**Vehicles Found:** {data['total']}")

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

# --- Page 2: Live Camera ---
elif page == "Live Camera":
    st.title("Live Camera Scan")
    st.markdown("Point your camera at a vehicle. Results will appear every 2 seconds.")

    # Load and embed the live camera HTML
    try:
        with open("frontend/live_camera.html", "r", encoding="utf-8") as f:
            html_string = f.read()

        # Substitute the configured backend URL into the placeholder
        html_string = html_string.replace(
            "API_URL_PLACEHOLDER",
            API_URL
        )

        components.html(html_string, height=700, scrolling=False)

    except FileNotFoundError:
        st.error("live_camera.html not found. Make sure it exists in the frontend/ folder.")