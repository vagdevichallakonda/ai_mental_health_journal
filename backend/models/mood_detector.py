import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from transformers import AutoTokenizer, TFAutoModel

# ---------------------------
# Attention Layer
# ---------------------------
class AttentionLayer(tf.keras.layers.Layer):
    """Additive attention over time dimension for sequences (batch, timesteps, features)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None
        self.b = None
        self.u = None
        self.supports_masking = True

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W = self.add_weight(name="attn_W",
                                 shape=(feature_dim, feature_dim),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="attn_b",
                                 shape=(feature_dim,),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="attn_u",
                                 shape=(feature_dim,),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        uit = tf.tensordot(x, self.W, axes=[2, 0]) + self.b
        uit = tf.tanh(uit)
        ait = tf.tensordot(uit, self.u, axes=[2, 0])
        if mask is not None:
            neg_inf = tf.constant(-1e9, dtype=ait.dtype)
            ait = tf.where(mask, ait, neg_inf)
        a = tf.nn.softmax(ait, axis=1)
        a = tf.expand_dims(a, axis=-1)
        output = tf.reduce_sum(x * a, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        return None

# ---------------------------
# Mapping 28 → 6 moods
# ---------------------------
BIG6_MAPPING = {
    # Anger
    'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
    'disgust': 'anger', 'remorse': 'anger',

    # Fear
    'fear': 'fear', 'nervousness': 'fear', 'embarrassment': 'fear',
    'grief': 'fear', 'confusion': 'fear',

    # Joy
    'joy': 'joy', 'amusement': 'joy', 'excitement': 'joy', 'gratitude': 'joy',
    'optimism': 'joy', 'pride': 'joy', 'relief': 'joy', 'approval': 'joy',

    # Surprise
    'surprise': 'surprise', 'realization': 'surprise', 'curiosity': 'surprise',

    # Love
    'love': 'love', 'caring': 'love', 'admiration': 'love', 'desire': 'love',

    # Sadness
    'sadness': 'sadness', 'disappointment': 'sadness', 'neutral': 'sadness'
}
BIG6_CLASSES = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# ---------------------------
# BiLSTM + Attention + BERT model
# ---------------------------
class BiLSTMAttentionMoodDetector:
    def __init__(self, max_len=128, lstm_units=64, bert_model_name='bert-base-uncased'):
        self.max_len = max_len
        self.lstm_units = lstm_units
        self.bert_model_name = bert_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = TFAutoModel.from_pretrained(
            self.bert_model_name,
            from_pt=False,   # Force TF-native checkpoint
            trainable=False
        )


        self.model: Model | None = None
        self.attention_model: Model | None = None
        self.label_encoder: LabelEncoder | None = None
        self.mood_categories = []  # will hold the 28 classes

    # ---------------------------
    # Data loading / preprocessing
    # ---------------------------
    def load_data(self):
        data_path = os.path.join("data", "mood_dataset.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Please place your dataset CSV there with columns 'text' and 'mood'."
            )

        df = pd.read_csv(data_path)
        required_cols = {'text', 'mood'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        if len(df) == 0:
            raise ValueError("Dataset is empty.")

        texts = df['text'].astype(str).values
        labels = df['mood'].astype(str).values
        return texts, labels

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='tf'
        )
        X = {'input_ids': encodings['input_ids'],
             'attention_mask': encodings['attention_mask']}

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        y = to_categorical(y_encoded, num_classes=num_classes)

        self.mood_categories = list(self.label_encoder.classes_)
        return X, y

    # ---------------------------
    # Model
    # ---------------------------
    def build_model(self):
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")

        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask)[0]

        lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.2))(bert_outputs)
        lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.2))(lstm1)

        attention_output = AttentionLayer()(lstm2)

        dense1 = Dense(128, activation='relu')(attention_output)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)

        output = Dense(len(self.mood_categories), activation='softmax')(dropout2)

        self.model = Model(inputs=[input_ids, attention_mask], outputs=output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                           loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    # ---------------------------
    # Train / Evaluate
    # ---------------------------
    def train(self, epochs=5, batch_size=16, validation_split=0.1):
        texts, labels = self.load_data()
        X, y = self.preprocess_data(texts, labels)
        self.build_model()
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history

    def evaluate_model(self):
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
            'classification_report': classification_report(true_classes, predicted_classes,
                                                           target_names=class_names, output_dict=True)
        }

    # ---------------------------
    # Predict
    # ---------------------------
    def predict(self, text):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        enc = self.tokenizer(texts,
                             truncation=True,
                             padding='max_length',
                             max_length=self.max_len,
                             return_tensors='tf')
        X = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}
        predictions = self.model.predict(X)
        probs_28 = np.array(predictions)

        results = []
        for row in probs_28:
            idx = int(np.argmax(row))
            primary_fine = self.label_encoder.inverse_transform([idx])[0]

            # Probabilities for each of the 28 moods
            prob_dict_28 = {m: float(row[i]) for i, m in enumerate(self.mood_categories)}

            # Aggregate to big6
            big6_probs = {c: 0.0 for c in BIG6_CLASSES}
            for mood28, prob in prob_dict_28.items():
                mapped_big = BIG6_MAPPING.get(mood28)
                if mapped_big in big6_probs:
                    big6_probs[mapped_big] += prob
            total = sum(big6_probs.values())
            if total > 0:
                big6_probs = {k: v/total for k, v in big6_probs.items()}

            primary_big = max(big6_probs, key=big6_probs.get)

            results.append({
                'primary_fine_mood': primary_fine,
                'confidence_fine': float(row[idx]),
                'primary_big6_mood': primary_big,
                'confidence_big6': float(big6_probs[primary_big]),
                'all_28_probs': prob_dict_28,
                'all_big6_probs': big6_probs
            })

        return results[0] if len(results) == 1 else results

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    detector = BiLSTMAttentionMoodDetector()
    print("Training Bi-LSTM + Attention mood detector with BERT embeddings...")
    detector.train(epochs=3)  # adjust as needed

    test_texts = [
        "I'm feeling really anxious about tomorrow's meeting",
        "Had such a wonderful day at the beach with family!",
        "Can't seem to get enough sleep, always tired",
        "Feeling hopeful about the new opportunities ahead",
    ]

    for text in test_texts:
        result = detector.predict(text)
        print(f"\nText: {text}")
        print("Fine-grained prediction:", result['primary_fine_mood'], "Conf:", result['confidence_fine'])
        print("Big-6 prediction:", result['primary_big6_mood'], "Conf:", result['confidence_big6'])
