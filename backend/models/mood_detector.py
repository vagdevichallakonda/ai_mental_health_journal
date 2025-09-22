import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Embedding
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


class AttentionLayer(tf.keras.layers.Layer):
    """Additive attention over time dimension for sequences (batch, timesteps, features)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None
        self.b = None
        self.u = None
        self.supports_masking = True

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        feature_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="attn_W",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attn_b",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attn_u",
            shape=(feature_dim,),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (batch, timesteps, features)
        uit = tf.tensordot(x, self.W, axes=[2, 0]) + self.b  # (batch, timesteps, features)
        uit = tf.tanh(uit)
        ait = tf.tensordot(uit, self.u, axes=[2, 0])         # (batch, timesteps)

        if mask is not None:
            # mask is boolean of shape (batch, timesteps)
            neg_inf = tf.constant(-1e9, dtype=ait.dtype)
            ait = tf.where(mask, ait, neg_inf)

        a = tf.nn.softmax(ait, axis=1)                       # (batch, timesteps)
        a = tf.expand_dims(a, axis=-1)                       # (batch, timesteps, 1)
        output = tf.reduce_sum(x * a, axis=1)                # (batch, features)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        # We reduce the time dimension, so no mask to propagate
        return None


class BiLSTMAttentionMoodDetector:
    """Bi-LSTM with Attention mechanism for mood detection"""

    def __init__(self, max_words=10_000, max_len=100, embedding_dim=100, lstm_units=64):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        # Model components
        self.model: Model | None = None
        self.attention_model: Model | None = None
        self.tokenizer: Tokenizer | None = None
        self.label_encoder: LabelEncoder | None = None

        # Mood categories
        self.mood_categories = ['tired', 'anxious', 'hopeful', 'sad', 'neutral', 'happy']

    # ---------------------------
    # Data loading / preprocessing
    # ---------------------------
    def load_data(self):
        """Load and prepare mood dataset. Requires an existing CSV at data/mood_dataset.csv."""
        data_path = os.path.join("data", "mood_dataset.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Please place your dataset CSV there with columns 'text' and 'mood'."
            )

        df = pd.read_csv(data_path)
        # Basic validations
        required_cols = {'text', 'mood'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}. Found columns: {list(df.columns)}")
        if len(df) == 0:
            raise ValueError("Dataset is empty. Please provide at least one row.")

        texts = df['text'].astype(str).values
        labels = df['mood'].astype(str).values
        return texts, labels

    def preprocess_data(self, texts, labels):
        """Preprocess text data and encode labels"""
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(y_encoded, num_classes=len(self.mood_categories))

        return X, y

    # ---------------------------
    # Model
    # ---------------------------
    def build_model(self):
        """Build Bi-LSTM model with attention mechanism"""
        input_layer = Input(shape=(self.max_len,))

        embedding = Embedding(
            input_dim=self.max_words,
            output_dim=self.embedding_dim,
            mask_zero=True
        )(input_layer)

        lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.2))(embedding)
        lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.2))(lstm1)

        attention_output = AttentionLayer()(lstm2)

        dense1 = Dense(128, activation='relu')(attention_output)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)

        output = Dense(len(self.mood_categories), activation='softmax')(dropout2)

        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Attention visualization model
        self.attention_model = Model(inputs=input_layer, outputs=attention_output)
        return self.model

    # ---------------------------
    # Train / Evaluate
    # ---------------------------
    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        print("Loading data...")
        texts, labels = self.load_data()

        print("Preprocessing data...")
        X, y = self.preprocess_data(texts, labels)

        print("Building model...")
        self.build_model()

        print("Training model...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        # Save model and components
        self.save_model('models/saved_models/mood_detector.pkl')

        print("Training completed!")
        return history

    def evaluate_model(self):
        """Evaluate model performance"""
        texts, labels = self.load_data()
        X, y = self.preprocess_data(texts, labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)

        class_names = self.label_encoder.classes_
        print("Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_names))

        print("\nConfusion Matrix:")
        print(confusion_matrix(true_classes, predicted_classes))

        return {
            'accuracy': float(np.mean(predicted_classes == true_classes)),
            'classification_report': classification_report(
                true_classes, predicted_classes, target_names=class_names, output_dict=True
            )
        }

    # ---------------------------
    # Inference
    # ---------------------------
    def predict(self, text):
        """Predict mood from text"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Re-train and save, then call load_model with the .pkl path.")
        if self.model is None:
            raise RuntimeError("Model is not loaded. Ensure *_model.keras exists and is loadable.")

        if isinstance(text, str):
            text = [text]

        sequences = self.tokenizer.texts_to_sequences(text)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        predictions = self.model.predict(X)                  # (N, C)
        probs = np.array(predictions).squeeze()              # -> (C,) if single, else (N, C)
        if predictions.shape[0] == 1:
            if probs.ndim != 1:
                raise RuntimeError(f"Unexpected probs shape after squeeze: {probs.shape}")
            if not np.issubdtype(probs.dtype, np.floating):
                probs = probs.astype(float)

            primary_mood_idx = int(np.argmax(probs))
            confidence = float(probs[primary_mood_idx])
            primary_mood = self.label_encoder.inverse_transform([primary_mood_idx])[0]

            all_moods = {}
            for mood in self.mood_categories:
                if mood in self.label_encoder.classes_:
                    mood_idx = list(self.label_encoder.classes_).index(mood)
                    all_moods[mood] = float(probs[mood_idx])

            return {
                'primary_mood': primary_mood,
                'confidence': confidence,
                'all_moods': all_moods
            }
        else:
            # Batch case: return list of dicts
            results = []
            for row in probs:
                row = row.astype(float)
                idx = int(np.argmax(row))
                primary_mood = self.label_encoder.inverse_transform([idx])[0]
                result = {
                    'primary_mood': primary_mood,
                    'confidence': float(row[idx]),
                    'all_moods': {m: float(row[list(self.label_encoder.classes_).index(m)]) for m in self.mood_categories if m in self.label_encoder.classes_}
                }
                results.append(result)
            return results

    def get_attention_weights(self, text):
        """Get attention weights for interpretability"""
        if self.tokenizer is None or self.attention_model is None:
            raise RuntimeError("Tokenizer/attention_model not loaded.")
        if isinstance(text, str):
            text = [text]

        sequences = self.tokenizer.texts_to_sequences(text)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        attention_output = self.attention_model.predict(X)
        return attention_output

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save_model(self, filepath):
        """Save model and components
        filepath: path to components pickle (e.g., models/saved_models/mood_detector.pkl)
        Also writes a sibling Keras model file: <base>_model.keras
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save Keras model (modern format)
        model_path = filepath.replace('.pkl', '_model.keras')
        if self.model is None:
            raise RuntimeError("Cannot save: model is None.")
        self.model.save(model_path)

        # Save other components
        components = {
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'max_words': self.max_words,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'mood_categories': self.mood_categories
        }

        with open(filepath, 'wb') as f:
            pickle.dump(components, f)

        print(f"Model saved to {filepath} and {model_path}")

    def load_model(self, filepath):
        """Load model and components"""
        print("[MoodDetector] load_model called with:", filepath)

        # Prefer .keras; fallback to .h5 if present
        model_path_keras = filepath.replace('.pkl', '_model.keras')
        model_path_h5 = filepath.replace('.pkl', '_model.h5')

        if os.path.exists(model_path_keras):
            model_path = model_path_keras
        elif os.path.exists(model_path_h5):
            model_path = model_path_h5
        else:
            raise FileNotFoundError(f"No model file found (.keras or .h5) for base: {filepath}")

        # Minimal custom objects; add more only if needed
        custom_objects = {
            'AttentionLayer': AttentionLayer,
        }

        try:
            self.model = keras_load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model from {model_path}: {e}")

        # Load components pickle
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Components pickle not found: {filepath}")
        if os.path.getsize(filepath) == 0:
            raise ValueError(f"Components pickle is empty: {filepath}. Re-train to regenerate.")

        try:
            with open(filepath, 'rb') as f:
                components = pickle.load(f)
        except EOFError:
            raise ValueError(f"Components pickle is corrupted or not a valid pickle: {filepath}. Re-train to regenerate.")

        self.tokenizer = components.get('tokenizer')
        self.label_encoder = components.get('label_encoder')
        self.max_words = components.get('max_words', self.max_words)
        self.max_len = components.get('max_len', self.max_len)
        self.embedding_dim = components.get('embedding_dim', self.embedding_dim)
        self.lstm_units = components.get('lstm_units', self.lstm_units)
        self.mood_categories = components.get('mood_categories', self.mood_categories)

        # Build attention submodel only if the model is loaded
        if self.model is not None:
            try:
                input_layer = self.model.input
                self.attention_model = None
                for layer in self.model.layers:
                    if isinstance(layer, AttentionLayer):
                        self.attention_model = Model(inputs=input_layer, outputs=layer.output)
                        break
                if self.attention_model is None:
                    print("[WARN] Attention submodel not constructed (layer not found). Predictions still work.")
            except Exception as e:
                print(f"[WARN] Failed to construct attention submodel: {e}")
                self.attention_model = None
        else:
            raise RuntimeError("self.model is None after loading; cannot build attention submodel.")

        # Final sanity checks
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded from pickle. Re-train and save again.")
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not loaded from pickle. Re-train and save again.")
        if self.max_len is None:
            raise RuntimeError("max_len missing in components.")

        print(f"[MoodDetector] Model and components loaded successfully from {filepath}")


if __name__ == "__main__":
    # Example usage
    detector = BiLSTMAttentionMoodDetector()

    # Train the model
    print("Training Bi-LSTM + Attention mood detector...")
    detector.train(epochs=30)

    # Test prediction
    test_texts = [
        "I'm feeling really anxious about tomorrow's meeting",
        "Had such a wonderful day at the beach with family!",
        "Can't seem to get enough sleep, always tired",
        "Feeling hopeful about the new opportunities ahead",
    ]

    for text in test_texts:
        result = detector.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted mood: {result['primary_mood']} (confidence: {result['confidence']:.2f})")
        print(f"All moods: {result['all_moods']}")

    # Evaluate model
    print("\n" + "=" * 50)
    print("Model Evaluation:")
    evaluation = detector.evaluate_model()
    print(f"Overall Accuracy: {evaluation['accuracy']:.4f}")
