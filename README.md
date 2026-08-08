# Tag — Vehicle Recognition System

A vehicle detection and license plate recognition system with a web dashboard and live camera support.

---

## Features

- **Vehicle Detection** — Identifies cars, trucks, motorcycles, buses, and bicycles using YOLO.
- **License Plate Reading** — Extracts plate text using EasyOCR, with contour-based plate cropping and a bottom-half fallback.
- **Web Dashboard** — Two modes: upload a photo, or scan in real time via camera.
- **Live Camera Support** — Captures a frame every 2 seconds and sends it to the backend; works on mobile devices over HTTPS.
- **FastAPI Backend** — Lightweight, performant, self-documenting via Swagger UI.
- **Self-Signed SSL** — Enables camera access (`getUserMedia`) on modern browsers over your local network.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI, Uvicorn |
| Vehicle Detection | YOLO (Ultralytics) |
| OCR | EasyOCR |
| Image Processing | OpenCV |
| Frontend | Streamlit, HTML, JavaScript |
| Camera Access | WebRTC (`getUserMedia`) |

---

## Prerequisites

- Python 3.8+
- A webcam-capable device for the Live Camera mode
- Two machines/devices need to be on the **same local network** if you're accessing the dashboard from a phone or another computer

---

## Project Structure

```
Tag/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Configuration settings
│   ├── models/
│   │   └── schemas.py           # API response models
│   ├── routes/
│   │   └── predict.py           # /predict endpoint
│   └── services/
│       └── vehicle_service.py   # Core detection logic
├── frontend/
│   ├── dashboard.py             # Streamlit master dashboard
│   ├── live_camera.html         # Camera page (embedded in dashboard)
│   └── web_app.py               # Standalone upload interface
├── models/
│   └── yolov8n.pt                # YOLO model file
├── cert.pem                      # SSL certificate (generated locally)
├── key.pem                       # SSL private key (generated locally)
├── generate_cert.py              # Certificate generation script
├── vehicle_detector.py           # Standalone test script
└── requirements.txt               # Python dependencies
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/sirblackbird2/Tag.git
cd Tag
```

### 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate an SSL certificate

```bash
python generate_cert.py
```

This creates `cert.pem` and `key.pem`, required for HTTPS (camera access won't work without it on most browsers).

### 5. Start the backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-certfile cert.pem --ssl-keyfile key.pem
```

### 6. Point the dashboard at your backend

In `frontend/dashboard.py`, update:

```python
API_URL = "https://192.168.1.8:8000"  # replace with your machine's IP
```

### 7. Start the dashboard

In a new terminal:

```bash
streamlit run frontend/dashboard.py
```

### 8. Open in browser

- Dashboard: `http://localhost:8501`
- Backend API docs: `https://<your-ip>:8000/docs`

---

## Configuration

Backend settings live in `app/config.py`:

| Variable | Description |
|----------|-------------|
| `YOLO_MODEL` | Path to the YOLO model file |
| `DETECTION_CONFIDENCE` | Minimum detection confidence (default `0.5`) |
| `OCR_LANG` | EasyOCR language code (default `"en"`) |

Frontend backend address is configured once, at the top of `frontend/dashboard.py`:

```python
API_URL = "https://192.168.1.8:8000"  # update this if your IP changes
```

This is the only place the address needs to change — `live_camera.html` receives it automatically at render time.

---

## API Reference

### `POST /predict`

Upload an image and receive vehicle detection results.

**Request:** `multipart/form-data` with a `file` field (jpg, jpeg, or png)

**Response:**

```json
{
  "vehicles": [
    {
      "type": "Car",
      "confidence": 0.85,
      "bbox": [140, 45, 586, 388],
      "plate": "B1970SSW"
    }
  ],
  "total": 1
}
```

### `GET /health`

Simple health check.

**Response:**

```json
{
  "status": "healthy",
  "service": "Vehicle Recognition"
}
```

---

## Troubleshooting

| Symptom | Likely Cause / Fix |
|---|---|
| Browser blocks camera access | You're not on HTTPS or `localhost`. Confirm the backend is running with `--ssl-certfile`/`--ssl-keyfile` and the URL starts with `https://`. |
| "Your connection is not private" warning | Expected with a self-signed cert. Click through the browser warning (or install `cert.pem` as a trusted local CA) to proceed. |
| Dashboard shows "Could not connect to backend" | Check the backend is running and that `API_URL` in `dashboard.py` matches your machine's current IP and port. |
| `live_camera.html not found` error | Confirm the file exists at `frontend/live_camera.html` relative to where you launch `streamlit run`. |
| No vehicles detected | Try a clearer, closer, or better-lit image; confirm `DETECTION_CONFIDENCE` in `app/config.py` isn't set too high. |

---

## Security Notes

- The self-signed certificate is for **local development only**. Use a certificate from a trusted CA before exposing this beyond your own network.
- `verify=False` in `dashboard.py` skips SSL certificate validation — acceptable for a self-signed cert on your own LAN, but should not be used if the backend is ever reachable outside it.
- The `/predict` endpoint has no authentication. Anyone on your network can submit images to it. Add an API key check before deploying more broadly.

---

## Use Cases

- Parking lot monitoring and management
- Building security and access control
- Mobile vehicle scanning apps
- Traffic monitoring and analysis
- License plate database building

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

---

## License

MIT License

---

## Author

GitHub: **[sirblackbird2](https://github.com/sirblackbird2)**

## Repository

[https://github.com/sirblackbird2/Tag](https://github.com/sirblackbird2/Tag)
