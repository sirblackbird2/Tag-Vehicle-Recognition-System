import base64

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
    page_title="Tag — Vehicle Recognition",
    page_icon="🚗",
    layout="wide"
)

# --- Styling ---
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }

    .tag-card {
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .tag-card h4 {
        margin: 0 0 0.6rem 0;
        font-size: 1.05rem;
    }
    .tag-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
        margin-right: 0.4rem;
    }
    .tag-row {
        font-size: 0.92rem;
        color: #333;
        margin: 0.15rem 0;
    }
    .tag-label {
        color: #777;
        font-weight: 500;
        display: inline-block;
        min-width: 90px;
    }
    .tag-plate {
        font-family: 'Consolas', monospace;
        background: #f4f4f4;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TYPE_COLORS = {
    "Car": "#2ecc71",
    "Motorcycle": "#f1c40f",
    "Bus": "#e74c3c",
    "Truck": "#9b59b6",
    "Bicycle": "#3498db",
}


def render_vehicle_card(i, vehicle):
    vtype = vehicle.get("type", "Unknown")
    color = TYPE_COLORS.get(vtype, "#95a5a6")
    brand = vehicle.get("brand") or "Unknown"
    confidence = vehicle.get("confidence", 0)
    plate = vehicle.get("plate")

    plate_html = (
        f'<span class="tag-plate">{plate}</span>'
        if plate else '<span style="color:#999;">Not detected</span>'
    )

    st.markdown(
        f"""
        <div class="tag-card">
            <h4>Vehicle {i}
                <span class="tag-badge" style="background:{color};">{vtype}</span>
            </h4>
            <div class="tag-row"><span class="tag-label">Brand</span>{brand}</div>
            <div class="tag-row"><span class="tag-label">Confidence</span>{confidence:.1%}</div>
            <div class="tag-row"><span class="tag-label">Plate</span>{plate_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Sidebar Navigation ---
st.sidebar.title("🚗 Tag")
st.sidebar.caption("Vehicle detection, brand ID & plate recognition")
page = st.sidebar.radio(
    "Choose a mode:",
    ["Upload Photo", "Live Camera"]
)

# --- Page 1: Upload Photo ---
if page == "Upload Photo":
    st.title("Upload a Vehicle Photo")
    st.markdown("Upload an image to detect vehicles, classify their brand, and read license plates.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([3, 2])

        with col1:
            image_slot = st.empty()
            image_slot.image(uploaded_file, caption="Uploaded Image", width='stretch')

        if st.button("Analyze", type="primary"):
            with st.spinner("Processing..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    # verify=False skips SSL cert validation — fine for a self-signed
                    # cert on your own LAN, but do not use this if the server is ever
                    # exposed beyond your local network.
                    response = requests.post(
                        PREDICT_URL,
                        params={"annotate": "true"},
                        files=files,
                        verify=False,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # Swap the plain upload for the annotated version,
                        # if the backend returned one.
                        annotated_b64 = data.get("annotated_image")
                        if annotated_b64:
                            with col1:
                                image_slot.image(
                                    base64.b64decode(annotated_b64),
                                    caption="Detected Vehicles",
                                    width="stretch",
                                )

                        with col2:
                            st.subheader("Detection Results")
                            st.write(f"**Vehicles Found:** {data['total']}")

                            if data["total"] == 0:
                                st.warning("No vehicles detected in this image.")
                            else:
                                for i, vehicle in enumerate(data["vehicles"], 1):
                                    render_vehicle_card(i, vehicle)
                    else:
                        st.error(f"Server error: {response.status_code}")

                except requests.exceptions.Timeout:
                    st.error("Request timed out. The backend may be under heavy load — try again.")
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