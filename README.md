> **Note:** the `.pth` model files are tracked with [Git LFS](https://git-lfs.com/). Run `git lfs install` once before cloning, or `git lfs pull` after cloning, to fetch the actual model weights.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/sirblackbird2/Tag-Vehicle-Recognition-System.git
cd Tag-Vehicle-Recognition-System
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
| `BRAND_MODEL_PATH` | Path to the car brand classifier weights |
| `BRAND_CONFIDENCE_THRESHOLD` | Min. softmax confidence to report a car brand instead of `"Unknown"` (default `0.65`) |
| `MOTORCYCLE_BRAND_MODEL_PATH` | Path to the motorcycle brand classifier weights |
| `MOTORCYCLE_BRAND_CONFIDENCE_THRESHOLD` | Min. softmax confidence to report a motorcycle brand instead of `"Unknown"` (default `0.65`) |

Frontend backend address is configured once, at the top of `frontend/dashboard.py`:

```python
API_URL = "https://192.168.1.8:8000"  # update this if your IP changes
```

This is the only place the address needs to change — `live_camera.html` receives it automatically at render time.

---

## Brand Classification

Each detected vehicle is routed to a brand classifier trained for its body type:

| Vehicle Type | Classifier | Brands |
|---|---|---|
| Motorcycle | `motorcycle_brand_classifier.pth` | Honda, Yamaha |
| Car, Bus, Truck, Bicycle | `brand_classifier.pth` | Toyota, Honda |

Both are ResNet18 models fine-tuned on cropped vehicle images. Predictions below their confidence threshold (see Configuration above) are returned as `"Unknown"` rather than a low-confidence guess. Honda appears in both classifiers since it makes both cars and motorcycles — each was trained separately on images of that body type so the model learns brand-specific cues (badge, grille/fairing design) rather than just body shape.

> **Note:** brand classification currently only distinguishes the trained classes above. Other brands will be misclassified as one of the trained options, or returned as `"Unknown"` if confidence is low — this reflects the current training data, not a general-purpose brand detector.

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
      "brand": "Toyota",
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
| Brand always shows `"Unknown"` or brand classifier fails to load | Confirm `git lfs pull` was run so `brand_classifier.pth`/`motorcycle_brand_classifier.pth` downloaded as real files, not LFS pointer stubs — check file size is tens of MB, not a few hundred bytes. |

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

[https://github.com/sirblackbird2/Tag-Vehicle-Recognition-System](https://github.com/sirblackbird2/Tag-Vehicle-Recognition-System)