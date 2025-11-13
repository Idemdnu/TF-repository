# TensorFlow Computer Vision Project

This is a repository originally focused on learning and using TensorFlow library with attempt to train and use the Computer Vision model.

Lately I used this repo for learning the Streamlit basics.

Lastly, I used remains of this repository to learn proper ML project structure, deployment and pipelies. I genuinelly hope this repo won't be modified again. But nobody knows :)

---

## Highlights
- **Transfer learning playground** – CIFAR-100 experiments built around EfficientNet variants (`notebooks/`) with reusable training utilities.
- **Model zoo on disk** – multiple `.keras`/`.h5` checkpoints under `weights/`, addressable via a single environment variable.
- **Streamlit inference UI** – `st_app.py` wraps MobileNetV2 preprocessing, ImageNet class decoding, and top‑5 predictions for any uploaded `.jpg/.png`.
- **Config-by-dotenv** – lightweight `.env` keeps secrets and weight selection out of the codebase.

---

## Architecture at a Glance
| Layer | Purpose | Key Files |
| --- | --- | --- |
| **Data + experimentation** | Load CIFAR-100, normalize, fine-tune EfficientNet backbones, and log metrics. | `notebooks/Cifar_100_classification.ipynb`, `notebooks/EfficientNetV2_implementation.ipynb` |
| **Model registry** | Store exported `.keras`/`.h5` checkpoints for reuse in notebooks or apps. | `weights/` |
| **Inference app** | Streamlit UI + TensorFlow runtime that loads the selected weight file and serves predictions. | `st_app.py`, `.env` |

---

## Repository Layout
```
.
├── st_app.py                      # Streamlit inference UI (MobileNetV2 + ImageNet labels)
├── requirements.txt               # Runtime dependencies for the app / notebooks
├── pyproject.toml                 # project metadata
├── data/
│   └── photo_test/                # Sample images used during experimentation
├── notebooks/
│   ├── Cifar_100_classification.ipynb     # Transfer-learning workflow on CIFAR-100
│   └── EfficientNetV2_implementation.ipynb# Prototype for EfficientNet-based prediction UI
├── weights/                       # Exported model checkpoints (see table below)
└── README_new.md
```

---

## Getting Started

### 1. Prerequisites
- Python **3.9+** (TensorFlow 2.16 wheels are validated up to Python 3.11).
- `pip` or [`uv`](https://docs.astral.sh/uv/) for dependency management.
- (Optional) A GPU-enabled TensorFlow install for faster notebook training.

### 2. Clone & Environment
```bash
git clone https://github.com/<your-username>/tensorflow_computer_vision_project.git
cd tensorflow_computer_vision_project
python3 -m venv .venv && source .venv/bin/activate  # or use `uv venv`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
# or, if you prefer uv:
uv pip sync requirements.txt
```

### 4. Environment Variables
Create a `.env` file (already tracked locally) with the weight file you want the app to load:
```
MODEL_WEIGHTS=mobile-net-v2-plain-model.keras
```
Any file listed under `weights/` can be selected here. Keep this file private if you add secrets or API keys later.

---

## Working with the Notebooks

### `notebooks/Cifar_100_classification.ipynb`
- Imports CIFAR-100 pickled archives, normalizes pixel values, and inspects random samples.
- Builds an **EfficientNetB0** backbone via `tensorflow.keras.applications`, freezes base layers, and adds task-specific dense heads.
- Trains for 10+ epochs using `ImageDataGenerator` pipelines and early experimentation-friendly hyperparameters.
- Saves the best-performing checkpoint to `weights/` for later reuse.

### `notebooks/EfficientNetV2_implementation.ipynb`
- Inspired by [EfficientNet V2 Streamlit tutorials](https://padhokshaja.medium.com/building-a-streamlit-app-to-identify-indian-food-part-ii-importing-efficientnet-v2-in-keras-9afca7364702).
- Demonstrates loading local test photos (`data/photo_test/`), resizing to `224x224`, batching, and scoring them with EfficientNetV2.
- Serves as a stepping stone toward the production Streamlit app found in `st_app.py`.

> 📓 Tip: Run notebooks inside the virtual environment so TensorFlow versions stay aligned with the app.

---

## Managing Model Weights

| File | Backbone | Source | Notes |
| --- | --- | --- | --- |
| `mobile-net-v2-plain-model.keras` | MobileNetV2 | Provided | Default weight for the Streamlit app (ImageNet classes). |
| `EffNetV2B0.keras` | EfficientNetV2-B0 | Custom | Exported from EfficientNet V2 notebook experiments. |
| `EffNet_V1.keras` / `.h5` | EfficientNetV1 | Custom | Legacy checkpoints kept for comparison. |

To switch models for the Streamlit app:
1. Copy or export your `.keras` file into `weights/`.
2. Update `MODEL_WEIGHTS` in `.env`.
3. Restart Streamlit (see below).

---

## Running the Streamlit App
```bash
streamlit run st_app.py --server.port 8501
```

What happens under the hood:
1. `st_app.py` loads environment variables (via `python-dotenv`) and fetches `weights/<MODEL_WEIGHTS>`.
2. Uploaded images are resized to **224×224**, converted to tensors, and preprocessed with `tensorflow.keras.applications.mobilenet_v2.preprocess_input`.
3. The MobileNetV2 model outputs logits that are decoded into ImageNet labels via `decode_predictions`.
4. The UI shows the uploaded image plus the **Top‑5 predictions** with confidence scores.

### Testing without the UI
You can quickly sanity-check a weight file in a Python shell:
```python
from PIL import Image
from st_app import preprocess_image, model
img = Image.open("data/photo_test/example.jpg")
probs = model.predict(preprocess_image(img))
```

---

## Deployment Notes
- **Streamlit Community Cloud**: push your repo, set the app entry point to `st_app.py`, and add `MODEL_WEIGHTS` to the secrets tab so Streamlit can read the correct filename.
- **Custom Hosting**: any environment with Python 3.9+, TensorFlow 2.16, and the weights folder available will run the app. For GPU-backed servers, ensure `tensorflow` is replaced with `tensorflow[and-cuda]`.
- **Large weights**: if you plan to store >100 MB models, consider using a remote object store (S3/GCS) and download the file at startup to avoid bloating the repo.

---

## Troubleshooting
- **`ModuleNotFoundError: tensorflow`** – reinstall inside the active virtual environment or match Python version (<=3.11 for TF 2.16.1).
- **Apple Silicon** – use `pip install tensorflow-macos tensorflow-metal` if the default wheel fails.
- **`FileNotFoundError: weights/<file>`** – confirm `.env` uses a filename that exists under `weights/`.
- **Slow first prediction** – TensorFlow lazily builds the graph; keep the app warm or invoke a dummy prediction during startup.

---

## Suggested Workflow
1. Explore CIFAR-100 or your custom dataset inside the notebooks.
2. Fine-tune EfficientNet/MobileNet, export the `.keras` file into `weights/`.
3. Update `.env` and validate predictions locally via Streamlit.
4. Deploy to Streamlit Community Cloud or your infra of choice.

---

## Roadmap Ideas
1. Add evaluation metrics/logging (TensorBoard or Weights & Biases) to notebooks.
2. Parameterize the Streamlit app so users can choose between available weight files at runtime.
3. Package preprocessing + inference as a standalone module for CLI batch scoring.
4. Automate tests that run a dummy inference to guard against breaking changes.

---

## Acknowledgements
- [TensorFlow](https://www.tensorflow.org/) for the core deep learning APIs and pretrained application models.
- [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) for the training data used in the notebooks.
- [Streamlit](https://streamlit.io/) for the rapid prototyping framework that powers the UI.

Happy experimenting! 🎉
