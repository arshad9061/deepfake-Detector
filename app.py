"""Fake / AI-generated detector for photos and videos (Gradio app).

Detection layers:
  1. Metadata check   - AI-tool markers (C2PA / OpenAI / Midjourney ...), camera EXIF
  2. Face-swap model  - runs ONLY on detected face crops
  3. AI-image models  - run on the whole image / frame (diffusion-style images)
  4. Screenshot check - warns when the input is a screenshot / graphic, not a photo
"""

import os
import re

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from transformers import pipeline

# ---------------------------------------------------------------- models ----
FACE_MODEL_ID = "dima806/deepfake_vs_real_image_detection"  # face-swap deepfakes
AI_MODEL_IDS = [  # AI-generated images (diffusion etc.)
    "Organika/sdxl-detector",
    "umm-maybe/AI-image-detector",
]

FAKE_TOKENS = {"fake", "artificial", "ai", "generated", "synthetic", "deepfake"}
REAL_TOKENS = {"real", "human", "natural", "authentic", "genuine"}


def load_model(model_id):
    try:
        print(f"Loading {model_id} ...")
        return pipeline("image-classification", model=model_id)
    except Exception as e:  # keep the app alive if one model fails to load
        print(f"[warn] could not load {model_id}: {e}")
        return None


face_clf = load_model(FACE_MODEL_ID)
ai_clfs = [(mid, m) for mid in AI_MODEL_IDS if (m := load_model(mid)) is not None]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def fake_probability(clf, img):
    """Probability (0..1) that `img` is fake/AI according to `clf`, or None."""
    if clf is None:
        return None
    results = clf(img, top_k=None)
    fake = real = 0.0
    seen_fake = seen_real = False
    for r in results:
        tokens = set(re.split(r"[^a-z0-9]+", r["label"].lower()))
        if tokens & FAKE_TOKENS:
            fake += float(r["score"])
            seen_fake = True
        elif tokens & REAL_TOKENS:
            real += float(r["score"])
            seen_real = True
    if seen_fake:
        return fake
    if seen_real:
        return 1.0 - real
    return None


def ai_probability(img):
    """Max fake score over all AI-image models + per-model breakdown."""
    per_model = {}
    for mid, m in ai_clfs:
        p = fake_probability(m, img)
        if p is not None:
            per_model[mid.split("/")[-1]] = p
    if not per_model:
        return None, per_model
    # min (not max/mean): "AI" only when ALL AI-image models agree
    return min(per_model.values()), per_model


# ------------------------------------------------------------------ face ----
def detect_face(img):
    """Return the largest face crop (with padding) or None."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.25 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, arr.shape[1]), min(y + h + pad, arr.shape[0])
    return Image.fromarray(arr[y0:y1, x0:x1])


# -------------------------------------------------------------- metadata ----
STRONG_AI_MARKERS = [
    b"openai", b"chatgpt", b"dall-e", b"midjourney", b"stable diffusion",
    b"stablediffusion", b"comfyui", b"adobe firefly", b"leonardo.ai",
    b"trainedalgorithmicmedia", b"negative prompt", b"sd-metadata",
]
C2PA_MARKERS = [b"c2pa", b"contentcredentials", b"jumbf"]


def read_head_tail(path, head=5 * 1024 * 1024, tail=2 * 1024 * 1024):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read(head)
        if size > head:
            f.seek(max(size - tail, head))
            data += f.read()
    return data.lower()


def inspect_metadata(path, pil_img=None):
    info = {"ai_markers": [], "c2pa": False, "camera": None, "software": None}
    try:
        data = read_head_tail(path)
    except OSError:
        return info
    info["ai_markers"] = sorted(
        {m.decode() for m in STRONG_AI_MARKERS if m in data}
    )
    info["c2pa"] = any(m in data for m in C2PA_MARKERS)
    if pil_img is not None:
        try:
            exif = pil_img.getexif()
            make, model, software = exif.get(271), exif.get(272), exif.get(305)
            if make or model:
                info["camera"] = f"{make or ''} {model or ''}".strip()
            if software:
                info["software"] = str(software)
                low = str(software).lower()
                for m in STRONG_AI_MARKERS:
                    if m.decode() in low and m.decode() not in info["ai_markers"]:
                        info["ai_markers"].append(m.decode())
        except Exception:
            pass
    return info


# ------------------------------------------------------------ screenshot ----
def looks_like_screenshot(img, has_camera):
    """Heuristic: lots of perfectly flat pixels + no camera EXIF."""
    if has_camera:
        return False
    a = np.array(img.convert("L"))
    a = a[:1200, :1200]
    flat = float(np.mean(a[:, 1:] == a[:, :-1]))
    return flat > 0.6


# --------------------------------------------------------------- verdicts ----
def combine(face_p, ai_p):
    # The face-swap model over-fires on phone selfies / compressed photos,
    # so it gets a small weight; the AI-image models carry the verdict.
    parts = [(w, v) for w, v in ((0.25, face_p), (0.75, ai_p)) if v is not None]
    if not parts:
        return None
    total_w = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total_w


def verdict_text(p):
    # These open models lean towards "fake", so the fake bar is high.
    if p >= 0.85:
        return "⚠️ **FAKE / AI-generated aakan sadhyatha kooduthal**"
    if p <= 0.5:
        return "✅ **ORIGINAL aakan sadhyatha kooduthal**"
    return "❓ **Urappillaa (uncertain)** - oru tool-ine matram vishwasikkaruthu"


def pct(v):
    return "-" if v is None else f"{v:.0%}"


def metadata_lines(meta):
    lines = []
    if meta["ai_markers"]:
        lines.append(
            "🚨 **AI tool-inte adayalam metadata-yil kandu:** "
            + ", ".join(meta["ai_markers"])
        )
    elif meta["c2pa"]:
        lines.append(
            "🔎 Content Credentials (C2PA) undu, pakshe AI marker illa "
            "(camera / editing tool aayirikkam)."
        )
    if meta["camera"]:
        lines.append(
            f"📷 Camera EXIF undu: {meta['camera']} (real photo-kku positive signal, "
            "pakshe metadata fake cheyyam)."
        )
    if not lines:
        lines.append(
            "ℹ️ Metadata-yil AI adayalam onnum kandilla "
            "(screenshot / WhatsApp / social media aakumbol metadata pokum)."
        )
    return lines


# --------------------------------------------------------------- analysis ----
def analyze_image(path):
    if not path:
        return None, "Oru photo upload cheyyuka."
    try:
        raw = Image.open(path)
        meta = inspect_metadata(path, raw)
        img = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as e:
        return None, f"Image vayikkan pattiyilla: {e}"

    face = detect_face(img)
    face_p = fake_probability(face_clf, face) if face is not None else None
    ai_p, per_model = ai_probability(img)
    screenshot = looks_like_screenshot(img, bool(meta["camera"]))

    rows = [
        "| Check | Fake probability |",
        "|---|---|",
        f"| Face-swap model {'(face kandu)' if face is not None else '(face kandilla - skip)'} | {pct(face_p)} |",
    ]
    for name, p in per_model.items():
        rows.append(f"| AI-image model: {name} | {pct(p)} |")

    notes = metadata_lines(meta)
    if face is None:
        notes.append("👤 Face illa, athu kondu face-swap model upayogichilla.")
    if screenshot:
        notes.append(
            "🖥️ Ithu **screenshot / graphic** aanennu thonnunnu. Detectors "
            "photos-inu vendi ullathaanu, athukondu result vishwasikkaruthu."
        )

    # ---- final verdict
    if meta["ai_markers"]:
        label = {"Fake / AI": 0.99, "Real": 0.01}
        head = "⚠️ **AI-generated (metadata proof undu)**"
    elif screenshot:
        label = None
        head = "🖥️ **Screenshot / graphic** - photo detection ivide reliable alla"
    else:
        p = combine(face_p, ai_p)
        if p is None:
            return None, "Detector models load aayilla. Terminal-ile error nokkuka."
        if meta["camera"]:
            # real camera EXIF present and no AI marker -> be extra cautious
            p *= 0.7
            notes.append("📷 Camera EXIF kondu fake score kurachu (EXIF fake cheyyam).")
        label = {"Fake / AI": p, "Real": 1 - p}
        head = verdict_text(p)
        if face_p is not None and ai_p is not None and abs(face_p - ai_p) > 0.5:
            notes.append(
                "⚖️ Randu detector-um thammil yojikkunnilla. Oral 'fake' ennum "
                "mattayal 'real' ennum paranju."
            )

    report = head + "\n\n" + "\n".join(rows) + "\n\n" + "\n\n".join(notes)
    return label, report


def analyze_video(path, n_frames=16):
    if not path:
        return None, "Oru video upload cheyyuka."
    meta = inspect_metadata(path)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return None, "Video vayikkan pattiyilla."

    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    face_scores, ai_scores = [], []
    used = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        used += 1
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = detect_face(img)
        if face is not None:
            fp = fake_probability(face_clf, face)
            if fp is not None:
                face_scores.append(fp)
        ap, _ = ai_probability(img)
        if ap is not None:
            ai_scores.append(ap)
    cap.release()
    if used == 0:
        return None, "Frames edukkan pattiyilla."

    face_p = float(np.mean(face_scores)) if face_scores else None
    ai_p = float(np.mean(ai_scores)) if ai_scores else None
    face_flag = float(np.mean(np.array(face_scores) > 0.5)) if face_scores else None
    ai_flag = float(np.mean(np.array(ai_scores) > 0.5)) if ai_scores else None

    rows = [
        "| Check | Average fake probability | Fake ennu thonniya frames |",
        "|---|---|---|",
        f"| Face-swap model ({len(face_scores)}/{used} frames-il face) | {pct(face_p)} | {pct(face_flag)} |",
        f"| AI-image models ({len(ai_scores)} frames) | {pct(ai_p)} | {pct(ai_flag)} |",
    ]
    notes = metadata_lines(meta)
    if not face_scores:
        notes.append("👤 Ethu frame-ilum face kandilla, face-swap result illa.")

    if meta["ai_markers"]:
        label = {"Fake / AI": 0.99, "Real": 0.01}
        head = "⚠️ **AI-generated (metadata proof undu)**"
    else:
        p = combine(face_p, ai_p)
        if p is None:
            return None, "Detector models load aayilla. Terminal-ile error nokkuka."
        label = {"Fake / AI": p, "Real": 1 - p}
        head = verdict_text(p)

    report = head + "\n\n" + "\n".join(rows) + "\n\n" + "\n\n".join(notes)
    return label, report


# --------------------------------------------------------------------- UI ----
with gr.Blocks(title="Fake Photo/Video Detector") as demo:
    gr.Markdown(
        "# 🕵️ Fake / AI Photo & Video Detector (v4)\n"
        "Photo or video upload cheyyuka. Result oru **probability** aanu, "
        "100% urappulla verdict alla. Puthiya AI generators ellam pidikkan pattilla."
    )
    with gr.Tab("Photo"):
        img_in = gr.Image(type="filepath", label="Photo")
        img_btn = gr.Button("Check cheyyuka", variant="primary")
        img_label = gr.Label(label="Result")
        img_report = gr.Markdown()
        img_btn.click(analyze_image, img_in, [img_label, img_report])
    with gr.Tab("Video"):
        vid_in = gr.Video(label="Video")
        vid_btn = gr.Button("Check cheyyuka", variant="primary")
        vid_label = gr.Label(label="Result")
        vid_report = gr.Markdown()
        vid_btn.click(analyze_video, vid_in, [vid_label, vid_report])

if __name__ == "__main__":
    demo.launch()
