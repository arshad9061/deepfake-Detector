# 🕵️ Fake / AI Photo & Video Detector

A simple web app that estimates whether an uploaded **photo or video** is real, a face-swap deepfake, or AI-generated. Built with Python, Gradio and Hugging Face models.

> ⚠️ **This tool gives probabilities, not proof.** No detector is 100% accurate. Do not use it as the only evidence for legal, journalistic or other high-stakes decisions.

## Features

- **Photo and video analysis** through a clean Gradio web UI
- **Metadata check** – looks for AI-tool markers (C2PA / Content Credentials, OpenAI, Midjourney, Stable Diffusion, Firefly, IPTC `trainedAlgorithmicMedia`) and camera EXIF
- **Face-swap deepfake model** – runs only on detected face crops
- **AI-generated image models** – run on the whole image / video frame
- **Screenshot / graphic detection** – warns when the input is not a real photo
- **Readable report** – per-model scores in a table, plus a note when detectors disagree
- **Video support** – samples 16 frames evenly and averages the scores

## How it works

| Layer | What it does |
|---|---|
| 1. Metadata | Scans the file for AI-generation markers and reads camera EXIF |
| 2. Face-swap model | `dima806/deepfake_vs_real_image_detection` on the detected face crop |
| 3. AI-image models | `Organika/sdxl-detector` and `umm-maybe/AI-image-detector` on the full image |
| 4. Screenshot check | Heuristic based on the share of perfectly flat pixels and missing camera EXIF |

**Verdict logic**

- AI marker found in metadata → **AI-generated** (strong evidence)
- Screenshot / graphic detected → no verdict (detectors are unreliable there)
- Otherwise a weighted score is computed: face-swap model 25 %, AI-image models 75 %. The AI-image score is the *lowest* of the AI models, so it only rises when all of them agree
- Score ≥ 85 % → likely fake / AI, ≤ 50 % → likely original, in between → uncertain
- If camera EXIF is present (and no AI marker), the fake score is reduced by 30 %

The thresholds are intentionally conservative because open models tend to over-flag real photos as fake.

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
gradio
transformers
torch
pillow
opencv-python<5
numpy
```

> `opencv-python<5` is required: OpenCV 5 no longer ships `cv2.CascadeClassifier`, which is used for face detection.

## Usage

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser, choose the **Photo** or **Video** tab, upload a file and click **Check**.

The first run downloads the models from Hugging Face (a few hundred MB), so an internet connection is needed once.

### Tips for better results

- Upload the **original file**. Screenshots and files re-sent through WhatsApp, Telegram or social media lose their metadata and are recompressed, which hurts detection.
- Photos with clearly visible faces work best for the face-swap check.
- For important cases, combine this tool with reverse image search (Google Lens, TinEye), source checking, and official detectors such as Google's SynthID Detector for Gemini images.

## Limitations

- **New generators are missed.** The free models were trained on older generators; images from the newest tools (e.g. recent GPT / Gemini / Midjourney versions) can be classified as real.
- **False positives happen.** Polished portraits, heavy filters, beauty modes, shallow depth of field and heavy compression can be flagged as fake.
- **Metadata is easy to lose or fake.** A missing marker proves nothing, and camera EXIF can be forged.
- **Face-swap model generalizes poorly** to phone selfies and compressed images, so it carries a low weight.
- **Screenshots and graphics** are not supported by the detectors.

## Project structure

```
.
├── app.py             # Gradio app + detection logic
├── requirements.txt   # Python dependencies
└── README.md
```

## Customization

- Swap or add models by editing `FACE_MODEL_ID` and `AI_MODEL_IDS` at the top of `app.py` (any Hugging Face `image-classification` model with real/fake-style labels works).
- Tune the weights in `combine()` and the thresholds in `verdict_text()` using your own set of known-real and known-AI images.
- Change the number of sampled video frames via the `n_frames` argument of `analyze_video()`.

## Roadmap

- [ ] Calibration script to tune weights/thresholds on your own real vs. AI images
- [ ] Audio deepfake (voice clone) detection
- [ ] Temporal / lip-sync analysis for videos
- [ ] Optional integration with a commercial detection API

## Disclaimer

This project is for educational and research purposes. The results are estimates produced by third-party machine-learning models and may be wrong. The authors are not responsible for decisions made based on its output.

## Credits

- [Gradio](https://gradio.app/), [Hugging Face Transformers](https://huggingface.co/docs/transformers), [OpenCV](https://opencv.org/)
- Models: `dima806/deepfake_vs_real_image_detection`, `Organika/sdxl-detector`, `umm-maybe/AI-image-detector`

## License

Add a license of your choice (e.g. MIT) in a `LICENSE` file.
