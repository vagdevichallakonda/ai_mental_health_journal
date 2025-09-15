"""
CNN Model for Voice Emotion Recognition
Processes audio using Mel spectrograms and 1D CNN for emotion classification
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import warnings
import json

warnings.filterwarnings('ignore')


class VoiceEmotionCNN:
    """CNN model for voice emotion recognition"""

    # Choose the on-disk format for the Keras model: "h5" (legacy) or "keras" (Keras v3 native)
    SAVE_FORMAT = "h5"  # change to "keras" if you want the new Keras v3 format

    def __init__(self, n_mfcc=13, n_chroma=12, n_spectral=7, sample_rate=22050, duration=3):
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_spectral = n_spectral
        self.sample_rate = sample_rate
        self.duration = duration

        # Total number of features
        self.n_features = n_mfcc + n_chroma + n_spectral  # 32 features total

        # Model components
        self.model = None
        self.label_encoder = None
        self.scaler = None

        # Emotion categories (compatible with RAVDESS dataset)
        self.emotion_categories = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    def extract_features(self, audio_path, offset=0.5):
        """Extract audio features from file"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration, offset=offset)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            # Extract Chroma features
            chroma = librosa.feature.chroma(y=audio, sr=sr, n_chroma=self.n_chroma)
            chroma_scaled = np.mean(chroma.T, axis=0)

            # Extract Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

            spectral_features = [
                float(np.mean(spectral_centroids)),
                float(np.var(spectral_centroids)),
                float(np.mean(spectral_rolloff)),
                float(np.var(spectral_rolloff)),
                float(np.mean(spectral_bandwidth)),
                float(np.var(spectral_bandwidth)),
                float(np.mean(zero_crossing_rate))
            ]

            features = np.hstack([mfccs_scaled, chroma_scaled, spectral_features])
            return features

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return np.zeros(self.n_features, dtype=np.float32)

    def extract_mel_spectrogram(self, audio_path, offset=0.5):
        """Extract mel spectrogram for 2D CNN (alternative approach)"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration, offset=offset)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            print(f"Error extracting mel spectrogram from {audio_path}: {str(e)}")
            return np.zeros((128, 130), dtype=np.float32)

    def load_data(self):
        """Load voice emotion dataset (creates synthetic data if not available)"""
        os.makedirs('data', exist_ok=True)
        csv_path = 'data/voice_emotions.csv'
        if not os.path.exists(csv_path):
            self._create_synthetic_data()

        df = pd.read_csv(csv_path)
        # Expect features column to be JSON-encoded lists; if not, try to parse robustly
        feats = []
        for v in df['features'].values:
            if isinstance(v, str):
                try:
                    feats.append(np.array(json.loads(v), dtype=np.float32))
                except Exception:
                    # fallback for legacy stringified Python lists
                    feats.append(np.array(eval(v), dtype=np.float32))
            else:
                feats.append(np.array(v, dtype=np.float32))
        return feats, df['emotion'].values

    def _create_synthetic_data(self):
        """Create synthetic voice emotion data for demonstration"""
        print("Creating synthetic voice emotion dataset...")
        np.random.seed(42)
        n_samples_per_emotion = 50
        rows = []

        emotion_templates = {
            'happy': {'mfcc_mean': 5, 'chroma_mean': 0.6, 'spectral_mean': 2000},
            'sad': {'mfcc_mean': -2, 'chroma_mean': 0.3, 'spectral_mean': 1200},
            'angry': {'mfcc_mean': 8, 'chroma_mean': 0.8, 'spectral_mean': 3000},
            'neutral': {'mfcc_mean': 0, 'chroma_mean': 0.5, 'spectral_mean': 1500},
            'fearful': {'mfcc_mean': 3, 'chroma_mean': 0.4, 'spectral_mean': 1800},
            'surprised': {'mfcc_mean': 6, 'chroma_mean': 0.7, 'spectral_mean': 2500},
            'calm': {'mfcc_mean': -1, 'chroma_mean': 0.4, 'spectral_mean': 1300},
            'disgust': {'mfcc_mean': 2, 'chroma_mean': 0.3, 'spectral_mean': 1600}
        }

        for emotion, template in emotion_templates.items():
            for _ in range(n_samples_per_emotion):
                mfcc_features = np.random.normal(template['mfcc_mean'], 2, self.n_mfcc)
                chroma_features = np.random.normal(template['chroma_mean'], 0.1, self.n_chroma)
                spectral_base = template['spectral_mean']
                spectral_features = [
                    spectral_base + np.random.normal(0, 200),
                    np.random.normal(100, 50),
                    spectral_base * 1.2 + np.random.normal(0, 300),
                    np.random.normal(150, 75),
                    spectral_base * 0.8 + np.random.normal(0, 200),
                    np.random.normal(80, 40),
                    np.random.normal(0.1, 0.05)
                ]
                features = np.hstack([mfcc_features, chroma_features, spectral_features]).astype(float)
                rows.append({
                    'features': json.dumps(features.tolist()),  # JSON-safe
                    'emotion': emotion
                })

        df = pd.DataFrame(rows)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/voice_emotions.csv', index=False)
        print(f"Created synthetic voice emotion dataset with {len(df)} samples")

    def preprocess_data(self, features, emotions):
        """Preprocess features and encode labels"""
        X = np.array(features, dtype=np.float32)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(emotions)
        y = to_categorical(y_encoded, num_classes=len(self.emotion_categories))
        return X_scaled, y

    def build_model(self, input_shape):
        """Build 1D CNN model for voice emotion recognition"""
        model = Sequential([
            tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

            Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),

            Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(0.4),

            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(256, activation='relu'),
            Dropout(0.3),

            Dense(128, activation='relu'),
            Dropout(0.2),

            Dense(len(self.emotion_categories), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, epochs=100, batch_size=32, validation_split=0.2, save_dir='models/saved_models'):
        """Train the CNN model"""
        print("Loading voice emotion data...")
        features, emotions = self.load_data()

        print("Preprocessing data...")
        X, y = self.preprocess_data(features, emotions)

        print("Building CNN model...")
        self.build_model(X.shape[1])

        print("Model Architecture:")
        self.model.summary()

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7, monitor='val_loss')
        ]

        print("Training CNN model...")
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Save model
        filename = f"voice_emotion_cnn.{self.SAVE_FORMAT}"
        out_path = os.path.join(save_dir, filename)
        self.save_model(out_path)

        print("Voice emotion CNN training completed!")
        return True

    def predict(self, features):
        """Predict emotion from audio features"""
        import numpy as np

        # Allow path input
        if isinstance(features, str):
            features = self.extract_features(features)

        # Ensure 2D float32 input
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale if available
        features_scaled = self.scaler.transform(features) if self.scaler is not None else features

        # Predict -> ensure NumPy array
        
        preds = self.model.predict(features_scaled)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = np.asarray(preds)

        # Normalize to 1D probs (num_classes,)
        # Common case: (1, num_classes) -> take first row
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds
        # Already 1D
        elif preds.ndim == 1:
            probs = preds
        else:
            # Handle object/nested arrays by flattening fully, then taking the last 8 values
            flat = np.array(list(np.ravel(preds)), dtype=np.float64)
            if flat.size < 8:
                raise ValueError(f"Unexpected prediction size: {flat.size}")
            probs = flat[-8:]

        # Ensure 1D float array
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs.ndim != 1:
            probs = probs.ravel()

        # Sanity: enforce length equals number of classes
        num_classes = len(self.emotion_categories)
        if probs.size != num_classes:
            # If sizes differ (e.g., label set mismatch), trim or pad zeros
            if probs.size > num_classes:
                probs = probs[:num_classes]
            else:
                probs = np.pad(probs, (0, num_classes - probs.size), mode='constant', constant_values=0.0)

        # Top-1
        primary_idx = int(np.argmax(probs))
        confidence = float(probs[primary_idx])

        # Map label
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            classes = list(self.label_encoder.classes_)
            if primary_idx < len(classes):
                primary_emotion = self.label_encoder.inverse_transform([primary_idx])[0]
            else:
                primary_emotion = self.emotion_categories[primary_idx]
        else:
            primary_emotion = self.emotion_categories[primary_idx]

        # Full distribution
        all_emotions = {}
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            classes = list(self.label_encoder.classes_)
            for i, emotion in enumerate(classes):
                if i < probs.shape[0]:
                    all_emotions[emotion] = float(probs[i])
        else:
            for i, emotion in enumerate(self.emotion_categories):
                if i < probs.shape:
                    all_emotions[emotion] = float(probs[i])

        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'all_emotions': all_emotions
        }



    def predict_from_file(self, audio_file_path):
        features = self.extract_features(audio_file_path)
        return self.predict(features)

    def _components_path_for(self, filepath):
        # components file pairs with the model file, regardless of extension
        root, ext = os.path.splitext(filepath)
        return f"{root}_components.pkl"

    def save_model(self, filepath):
        """Save CNN model and components"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save Keras model
        if filepath.endswith('.h5'):
            self.model.save(filepath)  # legacy HDF5
        elif filepath.endswith('.keras'):
            self.model.save(filepath)  # Keras v3 format
        else:
            # default to chosen SAVE_FORMAT
            filepath = filepath + ('.h5' if self.SAVE_FORMAT == 'h5' else '.keras')
            self.model.save(filepath)

        # Save components
        components_path = self._components_path_for(filepath)
        components = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'n_mfcc': self.n_mfcc,
            'n_chroma': self.n_chroma,
            'n_spectral': self.n_spectral,
            'n_features': self.n_features,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'emotion_categories': self.emotion_categories
        }
        with open(components_path, 'wb') as f:
            pickle.dump(components, f)
        print(f"Voice emotion CNN model saved to {filepath}")
        print(f"Components saved to {components_path}")

    def load_model(self, filepath):
        """Load CNN model and components"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Only accept .h5 or .keras for Keras 3
        if not (filepath.endswith('.h5') or filepath.endswith('.keras')):
            raise ValueError("Unsupported model file extension. Use a .h5 or .keras file.")

        self.model = keras_load_model(filepath)

        components_path = self._components_path_for(filepath)
        if os.path.exists(components_path):
            with open(components_path, 'rb') as f:
                components = pickle.load(f)
            self.label_encoder = components.get('label_encoder', None)
            self.scaler = components.get('scaler', None)
            self.n_mfcc = components.get('n_mfcc', self.n_mfcc)
            self.n_chroma = components.get('n_chroma', self.n_chroma)
            self.n_spectral = components.get('n_spectral', self.n_spectral)
            self.n_features = components.get('n_features', self.n_features)
            self.sample_rate = components.get('sample_rate', self.sample_rate)
            self.duration = components.get('duration', self.duration)
            self.emotion_categories = components.get('emotion_categories', self.emotion_categories)

        print(f"Voice emotion CNN model loaded from {filepath}")

    def retrain(self):
        print("Retraining voice emotion CNN with new data...")
        return self.train()

    def evaluate_model(self):
        features, emotions = self.load_data()
        X, y = self.preprocess_data(features, emotions)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        class_names = self.label_encoder.classes_ if (self.label_encoder is not None and hasattr(self.label_encoder, 'classes_')) else self.emotion_categories
        print("Voice Emotion Recognition - Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_names))

        print("\nConfusion Matrix:")
        print(confusion_matrix(true_classes, predicted_classes))

        return {
            'accuracy': float(np.mean(predicted_classes == true_classes)),
            'classification_report': classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
        }

    def create_mel_spectrogram_cnn(self):
        """Alternative: Create 2D CNN for mel spectrograms"""
        model = Sequential([
            tf.keras.layers.Reshape((128, 130, 1), input_shape=(128, 130)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.emotion_categories), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
if __name__ == "__main__":
    # Example usage
    voice_cnn = VoiceEmotionCNN()

    # Train the model
    print("Training Voice Emotion CNN...")
    voice_cnn.train(epochs=50)

    # Test prediction with synthetic features
    print("\n" + "=" * 50)
    print("Testing Voice Emotion Recognition")
    print("=" * 50)

    test_emotions = ['happy', 'sad', 'angry', 'neutral']
    for emotion in test_emotions:
        if emotion == 'happy':
            test_features = np.random.normal(5, 2, voice_cnn.n_features)
        elif emotion == 'sad':
            test_features = np.random.normal(-2, 2, voice_cnn.n_features)
        elif emotion == 'angry':
            test_features = np.random.normal(8, 3, voice_cnn.n_features)
        else:  # neutral
            test_features = np.random.normal(0, 2, voice_cnn.n_features)

        result = voice_cnn.predict(test_features)
        print(f"\nTest emotion: {emotion}")
        print(f"Predicted emotion: {result['primary_emotion']} (confidence: {result['confidence']:.2f})")
        print("All emotions:", {k: f"{v:.2f}" for k, v in result['all_emotions'].items()})

    # Evaluate model
    print("\n" + "=" * 50)
    print("Model Evaluation:")
    evaluation = voice_cnn.evaluate_model()
    print(f"Overall Accuracy: {evaluation['accuracy']:.4f}")