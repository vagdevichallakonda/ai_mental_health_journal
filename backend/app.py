from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid
from config import HOST, PORT, DEBUG, UPLOAD_FOLDER, MOOD_DETECTOR_PATH, VAE_GENERATOR_PATH, VOICE_CNN_PATH
from models.mood_detector import BiLSTMAttentionMoodDetector
from models.vae_tip_generator import VAEMotivationalTipGenerator
from models.voice_emotion_cnn import VoiceEmotionCNN
from models.hybrid_system import HybridRuleMLSystem

app = Flask(__name__)
CORS(app)

# Load models
mood_detector = BiLSTMAttentionMoodDetector()
mood_detector.load_model(MOOD_DETECTOR_PATH)

vae_generator = VAEMotivationalTipGenerator()
vae_generator.load_model(VAE_GENERATOR_PATH)

voice_cnn = VoiceEmotionCNN()
voice_cnn.load_model(VOICE_CNN_PATH)

hybrid_system = HybridRuleMLSystem(mood_detector, vae_generator)

EMOTION_TO_MOOD_MAP = {
    'neutral': 'neutral', 'calm': 'neutral', 'happy': 'happy',
    'sad': 'sad', 'angry': 'anxious', 'fearful': 'anxious',
    'disgust': 'sad', 'surprised': 'hopeful'
}

def allowed_audio_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"wav", "mp3", "ogg"}


@app.route("/predict_mood", methods=["POST"])
def predict_mood():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Text input required"}), 400
    result = mood_detector.predict(data["text"])
    tips = hybrid_system.get_tips(result["primary_mood"], result["confidence"])
    return jsonify({"mood_result": result, "tips": tips})

@app.route("/predict_voice_emotion", methods=["POST"])
def predict_voice():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not allowed_audio_file(file.filename):
        return jsonify({"error": "Invalid audio format"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = voice_cnn.predict_from_file(filepath)
    os.remove(filepath)

    mapped_mood = EMOTION_TO_MOOD_MAP.get(result["primary_emotion"], "neutral")
    tips = hybrid_system.get_tips(mapped_mood, result["confidence"])

    return jsonify({"voice_result": result, "mapped_mood": mapped_mood, "tips": tips})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI Mental Health Journal Assistant API Running"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)