import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# The loader derives sibling model file from this .pkl:
# mood_detector.pkl -> mood_detector_model.keras (preferred) or _model.h5 (fallback)
MOOD_DETECTOR_PATH = os.path.join(MODEL_DIR, "mood_detector.pkl")

# Keep these for your other models (adjust if their loaders differ)
VAE_GENERATOR_PATH = os.path.join(MODEL_DIR, "vae_tip_generator.pkl")
VOICE_CNN_PATH = os.path.join(MODEL_DIR, "voice_emotion_cnn.h5")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "ogg"}
