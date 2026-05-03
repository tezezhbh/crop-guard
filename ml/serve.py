"""
serve.py — CropGuard AI FastAPI inference server
Mekelle Institute of Technology 2026

Run: uvicorn serve:app --host 127.0.0.1 --port 8000

Improvements over v1:
  - Crashes on class index mismatch (no silent wrong predictions)
  - Confidence threshold rejection (no diagnosis below MIN_CONFIDENCE)
  - Image quality validation (blur + resolution check)
  - Farmer-friendly disease names in response
  - Enriched response: disease_type, urgency, organic_alternative
  - Consistent PIL.BILINEAR interpolation matching training pipeline
  - File size guard at FastAPI layer (defence-in-depth)
  - CORS restricted to Node backend origin only
"""

import io
import json
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass  # dotenv optional in production
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_PATH       = Path(os.getenv("MODEL_PATH",       "saved_model/plant_disease_model"))
CLASS_INDEX_PATH = Path(os.getenv("CLASS_INDEX_PATH", "saved_model/plant_disease_model/variables/class_index.json"))
IMG_SIZE         = (224, 224)
MAX_FILE_BYTES   = int(os.getenv("MAX_FILE_BYTES", 10 * 1024 * 1024))

# Minimum confidence to return a diagnosis.
# Below this we ask the farmer to retake the photo.
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", 60.0))

# Minimum image resolution in pixels (width or height)
MIN_RESOLUTION   = int(os.getenv("MIN_RESOLUTION", 100))

# Minimum Laplacian variance to reject blurry images (lower = blurrier)
MIN_BLUR_SCORE   = float(os.getenv("MIN_BLUR_SCORE", 50.0))

# ── Load class index ───────────────────────────────────────────────────────
if not CLASS_INDEX_PATH.exists():
    print(f"[serve] FATAL: class_index.json not found at {CLASS_INDEX_PATH}", file=sys.stderr)
    sys.exit(1)

with open(CLASS_INDEX_PATH) as f:
    _raw = json.load(f)

class_index: dict[str, str] = {k: v for k, v in _raw.items() if not k.startswith("_")}
NUM_CLASSES = len(class_index)
print(f"[serve] Class index loaded: {NUM_CLASSES} classes from {CLASS_INDEX_PATH}")

# ── Load model ─────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print(f"[serve] FATAL: model not found at {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

print(f"[serve] Loading model from {MODEL_PATH}...")
_model  = tf.saved_model.load(str(MODEL_PATH))
_infer  = _model.signatures["serving_default"]
OUTPUT_KEY = list(_infer.structured_outputs.keys())[0]
print(f"[serve] Model output key: '{OUTPUT_KEY}'")

# ── STARTUP VALIDATION — crash loudly on mismatch ─────────────────────────
_dummy   = tf.zeros([1, *IMG_SIZE, 3], dtype=tf.float32)
_out_dim = int(_infer(_dummy)[OUTPUT_KEY].shape[-1])

if _out_dim != NUM_CLASSES:
    print(
        f"\n[serve] FATAL: model outputs {_out_dim} classes but "
        f"class_index.json has {NUM_CLASSES} entries.\n"
        f"  Fix: ensure saved_model and class_index.json come from the same training run.\n",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[serve] Startup validation passed — {NUM_CLASSES} classes match.")

# ── Farmer-friendly disease names ─────────────────────────────────────────
# Maps raw PlantVillage labels to clean names farmers can read
FRIENDLY_NAMES: dict[str, str] = {
    "Apple___Apple_scab":                                   "Apple Scab",
    "Apple___Black_rot":                                    "Apple Black Rot",
    "Apple___Cedar_apple_rust":                             "Cedar Apple Rust",
    "Apple___healthy":                                      "Apple — Healthy",
    "Blueberry___healthy":                                  "Blueberry — Healthy",
    "Cherry_(including_sour)___Powdery_mildew":             "Cherry Powdery Mildew",
    "Cherry_(including_sour)___healthy":                    "Cherry — Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":   "Maize Gray Leaf Spot",
    "Corn_(maize)___Common_rust_":                          "Maize Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight":                  "Maize Northern Leaf Blight",
    "Corn_(maize)___healthy":                               "Maize — Healthy",
    "Grape___Black_rot":                                    "Grape Black Rot",
    "Grape___Esca_(Black_Measles)":                         "Grape Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":           "Grape Leaf Blight",
    "Grape___healthy":                                      "Grape — Healthy",
    "Orange___Haunglongbing_(Citrus_greening)":             "Citrus Greening (HLB)",
    "Peach___Bacterial_spot":                               "Peach Bacterial Spot",
    "Peach___healthy":                                      "Peach — Healthy",
    "Pepper,_bell___Bacterial_spot":                        "Bell Pepper Bacterial Spot",
    "Pepper,_bell___healthy":                               "Bell Pepper — Healthy",
    "Potato___Early_blight":                                "Potato Early Blight",
    "Potato___Late_blight":                                 "Potato Late Blight",
    "Potato___healthy":                                     "Potato — Healthy",
    "Raspberry___healthy":                                  "Raspberry — Healthy",
    "Soybean___healthy":                                    "Soybean — Healthy",
    "Squash___Powdery_mildew":                              "Squash Powdery Mildew",
    "Strawberry___Leaf_scorch":                             "Strawberry Leaf Scorch",
    "Strawberry___healthy":                                 "Strawberry — Healthy",
    "Tomato___Bacterial_spot":                              "Tomato Bacterial Spot",
    "Tomato___Early_blight":                                "Tomato Early Blight",
    "Tomato___Late_blight":                                 "Tomato Late Blight",
    "Tomato___Leaf_Mold":                                   "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot":                          "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":        "Tomato Spider Mites",
    "Tomato___Target_Spot":                                 "Tomato Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":               "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus":                         "Tomato Mosaic Virus",
    "Tomato___healthy":                                     "Tomato — Healthy",
}

# ── Disease metadata (type, urgency, organic alternative) ─────────────────
DISEASE_META: dict[str, dict] = {
    "Apple___Apple_scab":
        {"type":"Fungal","urgency":"moderate","organic":"Apply sulfur spray or neem oil at first sign. Remove affected leaves."},
    "Apple___Black_rot":
        {"type":"Fungal","urgency":"high","organic":"Prune infected wood. Apply copper-based fungicide. Remove mummified fruit."},
    "Apple___Cedar_apple_rust":
        {"type":"Fungal","urgency":"moderate","organic":"Remove nearby cedar/juniper hosts. Apply sulfur fungicide."},
    "Apple___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed. Maintain orchard hygiene."},
    "Blueberry___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Cherry_(including_sour)___Powdery_mildew":
        {"type":"Fungal","urgency":"moderate","organic":"Apply potassium bicarbonate or neem oil. Improve air circulation."},
    "Cherry_(including_sour)___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        {"type":"Fungal","urgency":"moderate","organic":"Rotate crops. Remove crop debris. Apply neem oil at early stages."},
    "Corn_(maize)___Common_rust_":
        {"type":"Fungal","urgency":"moderate","organic":"Plant resistant varieties. Neem oil at early stage."},
    "Corn_(maize)___Northern_Leaf_Blight":
        {"type":"Fungal","urgency":"high","organic":"Rotate crops. Remove infected debris. Copper spray at early signs."},
    "Corn_(maize)___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Grape___Black_rot":
        {"type":"Fungal","urgency":"high","organic":"Remove mummified berries. Apply copper fungicide. Improve canopy airflow."},
    "Grape___Esca_(Black_Measles)":
        {"type":"Fungal","urgency":"high","organic":"No cure. Remove infected vines. Seal pruning wounds with paste."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        {"type":"Fungal","urgency":"moderate","organic":"Apply copper spray. Remove infected leaves. Improve airflow."},
    "Grape___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Orange___Haunglongbing_(Citrus_greening)":
        {"type":"Bacterial","urgency":"critical","organic":"No cure. Remove infected tree immediately. Control psyllid insects."},
    "Peach___Bacterial_spot":
        {"type":"Bacterial","urgency":"moderate","organic":"Apply copper bactericide. Avoid overhead irrigation."},
    "Peach___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Pepper,_bell___Bacterial_spot":
        {"type":"Bacterial","urgency":"moderate","organic":"Apply copper spray. Remove infected leaves. Use disease-free transplants."},
    "Pepper,_bell___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Potato___Early_blight":
        {"type":"Fungal","urgency":"moderate","organic":"Remove lower infected leaves. Apply copper-based fungicide. Rotate crops."},
    "Potato___Late_blight":
        {"type":"Fungal","urgency":"critical","organic":"⚠️ Act immediately. Remove all infected material. Copper spray. Destroy infected plants."},
    "Potato___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed. Scout regularly during wet weather."},
    "Raspberry___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Soybean___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Squash___Powdery_mildew":
        {"type":"Fungal","urgency":"low","organic":"Apply baking soda spray (1 tsp/L water) or neem oil. Improve airflow."},
    "Strawberry___Leaf_scorch":
        {"type":"Fungal","urgency":"moderate","organic":"Remove infected leaves. Apply copper fungicide. Avoid wetting foliage."},
    "Strawberry___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
    "Tomato___Bacterial_spot":
        {"type":"Bacterial","urgency":"moderate","organic":"Apply copper bactericide. Remove infected leaves. Avoid wet foliage."},
    "Tomato___Early_blight":
        {"type":"Fungal","urgency":"moderate","organic":"Remove lower infected leaves. Apply copper fungicide or neem oil. Mulch base."},
    "Tomato___Late_blight":
        {"type":"Fungal","urgency":"critical","organic":"⚠️ Act immediately. Remove infected parts. Apply copper spray. Destroy severely infected plants."},
    "Tomato___Leaf_Mold":
        {"type":"Fungal","urgency":"moderate","organic":"Improve ventilation. Reduce humidity. Apply copper or sulfur spray."},
    "Tomato___Septoria_leaf_spot":
        {"type":"Fungal","urgency":"moderate","organic":"Remove infected leaves. Apply copper spray. Mulch soil to reduce splash."},
    "Tomato___Spider_mites Two-spotted_spider_mite":
        {"type":"Pest","urgency":"moderate","organic":"Spray with neem oil or insecticidal soap. Increase humidity. Introduce predatory mites."},
    "Tomato___Target_Spot":
        {"type":"Fungal","urgency":"moderate","organic":"Remove heavily infected leaves. Apply neem oil. Stake plants for airflow."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        {"type":"Viral","urgency":"critical","organic":"No cure. Remove infected plants immediately. Use reflective mulch to deter whiteflies."},
    "Tomato___Tomato_mosaic_virus":
        {"type":"Viral","urgency":"high","organic":"No cure. Remove infected plants. Disinfect tools with bleach. Wash hands before handling."},
    "Tomato___healthy":
        {"type":"None","urgency":"none","organic":"No treatment needed."},
}

# ── Treatment recommendations ──────────────────────────────────────────────
RECOMMENDATIONS: dict[str, str] = {
    "Apple___Apple_scab":
        "Apply fungicide (Captan or Mancozeb) at green tip stage. Remove and destroy fallen infected leaves. Prune for good air circulation.",
    "Apple___Black_rot":
        "Prune out infected cankers and mummified fruit. Apply Captan or Thiophanate-methyl fungicide. Remove all infected material from the orchard.",
    "Apple___Cedar_apple_rust":
        "Apply myclobutanil or propiconazole fungicide starting at pink bud stage. Remove nearby cedar/juniper trees if possible.",
    "Apple___healthy":
        "No disease detected. Maintain regular pruning, balanced fertilisation, and monitor for early signs of pests or disease.",
    "Blueberry___healthy":
        "No disease detected. Maintain proper soil pH (4.5–5.5), adequate irrigation, and annual pruning.",
    "Cherry_(including_sour)___Powdery_mildew":
        "Apply sulphur-based or potassium bicarbonate fungicide. Improve air circulation by pruning. Avoid excessive nitrogen fertilisation.",
    "Cherry_(including_sour)___healthy":
        "No disease detected. Continue regular monitoring and maintain orchard hygiene.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Apply Azoxystrobin or Propiconazole at early signs. Use resistant hybrids in future seasons. Rotate crops.",
    "Corn_(maize)___Common_rust_":
        "Apply Mancozeb or Chlorothalonil if infection is severe. Plant resistant hybrids. Monitor fields during cool wet periods.",
    "Corn_(maize)___Northern_Leaf_Blight":
        "Apply Mancozeb or Azoxystrobin fungicide. Practice crop rotation. Bury or remove infected crop residue after harvest.",
    "Corn_(maize)___healthy":
        "No disease detected. Continue monitoring and maintain good agronomic practices.",
    "Grape___Black_rot":
        "Apply Mancozeb or Myclobutanil from bud break through harvest. Remove mummified berries and infected canes.",
    "Grape___Esca_(Black_Measles)":
        "No effective cure. Remove and destroy severely infected vines. Protect pruning wounds with fungicide paste.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Apply copper-based fungicide or Mancozeb. Improve air circulation. Remove infected leaves promptly.",
    "Grape___healthy":
        "No disease detected. Maintain canopy management and balanced nutrition.",
    "Orange___Haunglongbing_(Citrus_greening)":
        "No cure. Remove and destroy infected trees immediately. Control the Asian citrus psyllid with insecticides. Plant certified disease-free trees.",
    "Peach___Bacterial_spot":
        "Apply copper-based bactericide in autumn and early spring. Avoid overhead irrigation. Remove infected twigs.",
    "Peach___healthy":
        "No disease detected. Continue regular monitoring, proper pruning, and balanced fertilisation.",
    "Pepper,_bell___Bacterial_spot":
        "Apply copper-based bactericide. Remove infected material. Avoid overhead irrigation. Use disease-free transplants.",
    "Pepper,_bell___healthy":
        "No disease detected. Maintain proper spacing for air circulation.",
    "Potato___Early_blight":
        "Apply Chlorothalonil or Mancozeb. Remove infected lower leaves. Ensure adequate potassium. Practice crop rotation.",
    "Potato___Late_blight":
        "⚠️ Apply Mancozeb or Cymoxanil IMMEDIATELY — late blight spreads very rapidly. Remove and destroy all infected plant material.",
    "Potato___healthy":
        "No disease detected. Scout fields regularly, especially during cool wet weather.",
    "Raspberry___healthy":
        "No disease detected. Maintain proper cane management and remove old canes after fruiting.",
    "Soybean___healthy":
        "No disease detected. Monitor for soybean rust during the growing season.",
    "Squash___Powdery_mildew":
        "Apply potassium bicarbonate, neem oil, or sulphur at first sign. Improve air circulation. Avoid wetting foliage.",
    "Strawberry___Leaf_scorch":
        "Apply Captan or Myclobutanil. Remove infected leaves. Ensure good drainage and air circulation.",
    "Strawberry___healthy":
        "No disease detected. Maintain proper bed renovation after harvest.",
    "Tomato___Bacterial_spot":
        "Apply copper-based bactericide. Remove infected leaves and stems. Avoid overhead irrigation.",
    "Tomato___Early_blight":
        "Apply Chlorothalonil or Mancozeb. Remove lower infected leaves. Stake plants for better air circulation.",
    "Tomato___Late_blight":
        "⚠️ Apply Mancozeb or Cymoxanil IMMEDIATELY — this disease spreads extremely fast. Remove infected parts.",
    "Tomato___Leaf_Mold":
        "Improve ventilation. Apply Chlorothalonil or copper fungicide. Avoid wetting leaves.",
    "Tomato___Septoria_leaf_spot":
        "Apply Chlorothalonil or Mancozeb. Remove infected lower leaves. Mulch around base to prevent soil splash.",
    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Apply Abamectin or Bifenazate miticide. Use neem oil as an organic option. Increase humidity.",
    "Tomato___Target_Spot":
        "Apply Chlorothalonil or Azoxystrobin. Remove heavily infected leaves. Improve air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "No cure. Remove and destroy infected plants immediately. Control the whitefly vector with sticky traps.",
    "Tomato___Tomato_mosaic_virus":
        "No cure. Remove infected plants. Disinfect tools with bleach solution. Wash hands before handling plants.",
    "Tomato___healthy":
        "No disease detected. Continue regular scouting and maintain good cultural practices.",
}

# ── Preprocessing — BILINEAR matches ImageDataGenerator interpolation ──────
def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    """
    PIL BILINEAR resize + /255 normalise.
    Matches ImageDataGenerator(interpolation='bilinear') used in training.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return tf.constant(arr[np.newaxis, ...])

# ── Image quality validation ───────────────────────────────────────────────
def check_image_quality(image_bytes: bytes) -> dict:
    """
    Returns {"ok": bool, "reason": str, "blur_score": float}.
    Checks minimum resolution and blur using Laplacian variance.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        w, h = img.size

        if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
            return {
                "ok": False,
                "reason": f"Image too small ({w}×{h}px). Please use a photo of at least {MIN_RESOLUTION}×{MIN_RESOLUTION} pixels.",
                "blur_score": 0.0,
            }

        # Laplacian variance — standard blur detection metric
        arr = np.array(img, dtype=np.float64)
        kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        lap = convolve(arr, kernel)
        blur_score = float(lap.var())

        if blur_score < MIN_BLUR_SCORE:
            return {
                "ok": False,
                "reason": "Image appears blurry. Please hold the camera steady and take a sharper photo.",
                "blur_score": round(blur_score, 2),
            }

        return {"ok": True, "reason": "ok", "blur_score": round(blur_score, 2)}

    except Exception as e:
        return {"ok": False, "reason": f"Could not read image: {e}", "blur_score": 0.0}

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="CropGuard AI — Inference Server",
    description=f"Plant disease detection ({NUM_CLASSES} classes). Internal service — not publicly exposed.",
    version="2.0.0",
    docs_url="/docs" if os.getenv("NODE_ENV") != "production" else None,
    redoc_url=None,
)

_allowed_origins = [
    os.getenv("BACKEND_ORIGIN", "http://127.0.0.1:3001"),
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ── Response schemas ───────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    # Core result
    disease:             str     # raw PlantVillage label
    disease_friendly:    str     # farmer-readable name
    confidence_pct:      float
    is_healthy:          bool

    # Enrichment
    disease_type:        str     # Fungal / Bacterial / Viral / Pest / None
    urgency:             str     # none / low / moderate / high / critical
    recommendation:      str     # chemical treatment
    organic_alternative: str     # organic / low-resource treatment

    # Alternatives
    top3:                list[dict]

    # Quality info
    image_quality_score: float

class RejectionResponse(BaseModel):
    error:               str
    reason:              str
    retake_photo:        bool
    image_quality_score: float

# ── POST /predict ──────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Content-type guard
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, or WebP).")

    contents = await file.read()

    # File size guard
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum {MAX_FILE_BYTES // (1024*1024)} MB.")

    # Image quality validation
    quality = check_image_quality(contents)
    if not quality["ok"]:
        return RejectionResponse(
            error="Image quality too low for reliable diagnosis.",
            reason=quality["reason"],
            retake_photo=True,
            image_quality_score=quality["blur_score"],
        )

    # Preprocess and infer
    try:
        tensor = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    raw   = _infer(tensor)
    probs = raw[OUTPUT_KEY].numpy()[0]

    top_idx    = int(np.argmax(probs))
    top_label  = class_index[str(top_idx)]
    confidence = float(probs[top_idx]) * 100.0

    # Confidence threshold — reject low-confidence predictions
    if confidence < MIN_CONFIDENCE:
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"disease": FRIENDLY_NAMES.get(class_index[str(int(i))], class_index[str(int(i))]),
             "confidence_pct": round(float(probs[int(i)]) * 100.0, 2)}
            for i in top3_idx
        ]
        return RejectionResponse(
            error=(
                f"Confidence too low ({confidence:.1f}%). "
                f"The model is unsure — the leaf may not be clearly visible, "
                f"or the disease may not be in the supported list."
            ),
            reason=(
                "Please retake the photo: ensure the leaf fills the frame, "
                "use good lighting, and focus on the most affected area."
            ),
            retake_photo=True,
            image_quality_score=quality["blur_score"],
        )

    # Build top-3 with friendly names
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "disease":        class_index[str(int(i))],
            "disease_friendly": FRIENDLY_NAMES.get(class_index[str(int(i))], class_index[str(int(i))]),
            "confidence_pct": round(float(probs[int(i)]) * 100.0, 2),
        }
        for i in top3_idx
    ]

    # Fetch enrichment metadata
    meta = DISEASE_META.get(top_label, {
        "type": "Unknown", "urgency": "moderate",
        "organic": "Consult your local agricultural extension officer.",
    })

    return PredictionResponse(
        disease=top_label,
        disease_friendly=FRIENDLY_NAMES.get(top_label, top_label),
        confidence_pct=round(confidence, 2),
        is_healthy="healthy" in top_label.lower(),
        disease_type=meta["type"],
        urgency=meta["urgency"],
        recommendation=RECOMMENDATIONS.get(top_label, "Consult your local agricultural extension officer."),
        organic_alternative=meta["organic"],
        top3=top3,
        image_quality_score=quality["blur_score"],
    )

# ── GET /health ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":          "ok",
        "classes":         NUM_CLASSES,
        "output_key":      OUTPUT_KEY,
        "min_confidence":  MIN_CONFIDENCE,
        "min_resolution":  MIN_RESOLUTION,
    }

# ── GET /classes ───────────────────────────────────────────────────────────
@app.get("/classes")
def get_classes():
    return {
        k: {"raw": v, "friendly": FRIENDLY_NAMES.get(v, v)}
        for k, v in class_index.items()
    }
