"""
Variational Autoencoder (VAE) for Motivational Tip Generation
Generates context-specific motivational tips based on mood and text input
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Layer, Embedding, LSTM,
    RepeatVector, TimeDistributed, Flatten, Concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class KLDivergenceLayer(Layer):
    """
    Adds KL divergence loss via add_loss inside the graph.
    Expects [z_mean, z_log_var] as inputs.
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl = tf.reduce_mean(kl) * -0.5
        self.add_loss(kl)
        return z_mean  # pass-through (value unused downstream)


class VAEMotivationalTipGenerator:
    """VAE model for generating motivational tips"""

    def __init__(self, latent_dim=64, max_length=50, vocab_size=5000):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Model components
        self.encoder: Model | None = None
        self.decoder: Model | None = None
        self.vae: Model | None = None
        self.tokenizer: Tokenizer | None = None

        # Mood-specific embeddings and mappings
        self.mood_embeddings = None  # dict with mood_to_idx, idx_to_mood, num_moods

        # Cache to diversify tips
        self.generated_tips_cache = set()

    # ---------------------------
    # Data
    # ---------------------------
    def load_data(self):
        """Load motivational tips dataset"""
        data_path = os.path.join('data', 'motivational_tips.csv')
        if not os.path.exists(data_path):
            self._create_sample_tips_dataset(data_path)

        df = pd.read_csv(data_path)
        tips = df['tip'].values
        moods = df['mood'].values if 'mood' in df.columns else ['neutral'] * len(tips)
        return tips, moods

    def _create_sample_tips_dataset(self, data_path):
        """Create sample motivational tips dataset"""
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        tips_data = [
            # Anxious
            ("Take three deep breaths and remind yourself that this feeling will pass", "anxious"),
            ("Break your worries down into smaller, manageable pieces", "anxious"),
            ("Practice the 5-4-3-2-1 grounding technique to center yourself", "anxious"),
            ("Remember that anxiety is temporary, but your strength is permanent", "anxious"),
            ("Focus on what you can control and let go of what you cannot", "anxious"),
            ("Your worries are not predictions of the future", "anxious"),
            ("Take one step at a time, you don't need to solve everything at once", "anxious"),
            ("Breathe in calm, breathe out tension", "anxious"),
            ("You have survived difficult times before, you can do it again", "anxious"),
            ("Ground yourself in the present moment, not future fears", "anxious"),
            # Happy
            ("Savor this beautiful moment and let joy fill your heart", "happy"),
            ("Share your happiness with someone you love", "happy"),
            ("Take a moment to appreciate all the good in your life", "happy"),
            ("Let this positive energy inspire you to spread kindness", "happy"),
            ("Document this happy moment to revisit when you need a boost", "happy"),
            ("Your joy is contagious, let it brighten others' days too", "happy"),
            ("Celebrate the small wins, they add up to big victories", "happy"),
            ("Gratitude turns what we have into enough", "happy"),
            ("Happiness is not a destination, it's a way of traveling", "happy"),
            ("Your smile is your superpower, use it generously", "happy"),
            # Sad
            ("It's okay to feel sad, your emotions are valid and important", "sad"),
            ("This difficult time will pass, like clouds moving across the sky", "sad"),
            ("Be gentle with yourself, treat yourself with compassion", "sad"),
            ("Reach out to someone who cares about you", "sad"),
            ("Small acts of self-care can make a big difference", "sad"),
            ("Your feelings are temporary visitors, not permanent residents", "sad"),
            ("Even in darkness, you carry a light within you", "sad"),
            ("Healing takes time, be patient with your process", "sad"),
            ("You are stronger than you know and braver than you feel", "sad"),
            ("Tomorrow is a new day with new possibilities", "sad"),
            # Tired
            ("Rest is not a reward for work completed, it's a necessity", "tired"),
            ("Listen to your body, it's telling you what you need", "tired"),
            ("Quality sleep is an investment in tomorrow's energy", "tired"),
            ("Take a power nap if possible, even 10-20 minutes helps", "tired"),
            ("Hydrate yourself, dehydration increases fatigue", "tired"),
            ("Step outside for fresh air and natural light", "tired"),
            ("Prioritize your tasks, not everything needs to be done today", "tired"),
            ("Your body needs rest to recharge and rebuild", "tired"),
            ("Be kind to yourself when energy levels are low", "tired"),
            ("Sometimes the most productive thing you can do is rest", "tired"),
            # Hopeful
            ("Hold onto hope, it's the anchor that keeps you steady", "hopeful"),
            ("Your optimism is a powerful force for positive change", "hopeful"),
            ("Channel this hope into action toward your goals", "hopeful"),
            ("Hope is the thing with feathers that perches in the soul", "hopeful"),
            ("Share your hope with others who need encouragement", "hopeful"),
            ("Plant seeds of possibility with your hopeful energy", "hopeful"),
            ("Hope combined with action creates miracles", "hopeful"),
            ("Your hope today creates tomorrow's reality", "hopeful"),
            ("Believe in the beauty of your dreams", "hopeful"),
            ("Hope is seeing light despite all the darkness", "hopeful"),
            # Neutral
            ("Take time to check in with yourself and your needs", "neutral"),
            ("Practice mindfulness and stay present in this moment", "neutral"),
            ("Set one small intention for the day ahead", "neutral"),
            ("Balance is not something you find, it's something you create", "neutral"),
            ("Progress is progress, no matter how small", "neutral"),
            ("Be curious about your inner world and experiences", "neutral"),
            ("Consistency in small actions leads to big changes", "neutral"),
            ("Your mental health matters, prioritize it daily", "neutral"),
            ("Notice the good around you, it's always there", "neutral"),
            ("Every day is a new opportunity to grow and learn", "neutral"),
        ]

        expanded_tips = []
        for tip, mood in tips_data:
            expanded_tips.append((tip, mood))
            # basic duplicates to expand dataset mildly
            expanded_tips.append((tip.lower(), mood))
            expanded_tips.append((tip.replace("you", "you"), mood))   # keep original
            expanded_tips.append((tip.replace("your", "your"), mood)) # keep original

        df = pd.DataFrame(expanded_tips, columns=['tip', 'mood'])
        df.to_csv(data_path, index=False)
        print(f"Created motivational tips dataset with {len(df)} entries")

    def preprocess_data(self, tips, moods):
        """Preprocess tips data"""
        # Tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(tips)

        sequences = self.tokenizer.texts_to_sequences(tips)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')

        # Mood mapping
        unique_moods = sorted(list(set(moods)))
        mood_to_idx = {m: i for i, m in enumerate(unique_moods)}
        mood_indices = np.array([mood_to_idx[m] for m in moods], dtype=np.int32)

        self.mood_embeddings = {
            'mood_to_idx': mood_to_idx,
            'idx_to_mood': {i: m for m, i in mood_to_idx.items()},
            'num_moods': len(unique_moods)
        }

        return X, mood_indices

    # ---------------------------
    # Model
    # ---------------------------
    def build_encoder(self):
        """Build VAE encoder"""
        # Inputs
        text_input = Input(shape=(self.max_length,), name="enc_text_in")
        mood_input = Input(shape=(1,), name="enc_mood_in")

        # Mood embedding
        mood_emb = Embedding(self.mood_embeddings['num_moods'], 16, name="mood_emb")(mood_input)
        mood_flat = Flatten(name="mood_flat")(mood_emb)

        # Text embedding and LSTM
        text_emb = Embedding(self.vocab_size, 128, mask_zero=True, name="text_emb")(text_input)
        lstm_out = LSTM(128, name="enc_lstm")(text_emb)

        # Concatenate
        combined = Concatenate(name="enc_concat")([lstm_out, mood_flat])

        # Hidden
        hidden = Dense(256, activation='relu', name="enc_dense1")(combined)
        hidden = Dense(128, activation='relu', name="enc_dense2")(hidden)

        # Latent params
        z_mean = Dense(self.latent_dim, name="z_mean")(hidden)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(hidden)

        # KL loss via layer
        _ = KLDivergenceLayer(name="kl_loss")([z_mean, z_log_var])

        # Sample
        z = Sampling(name="z")([z_mean, z_log_var])

        encoder = Model([text_input, mood_input], [z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self):
        """Build VAE decoder"""
        latent_input = Input(shape=(self.latent_dim,), name="dec_z_in")
        mood_input = Input(shape=(1,), name="dec_mood_in")

        mood_emb = Embedding(self.mood_embeddings['num_moods'], 16, name="dec_mood_emb")(mood_input)
        mood_flat = Flatten(name="dec_mood_flat")(mood_emb)

        combined = Concatenate(name="dec_concat")([latent_input, mood_flat])
        hidden = Dense(128, activation='relu', name="dec_dense1")(combined)
        hidden = Dense(256, activation='relu', name="dec_dense2")(hidden)

        repeated = RepeatVector(self.max_length, name="dec_repeat")(hidden)
        lstm_out = LSTM(128, return_sequences=True, name="dec_lstm")(repeated)
        output = TimeDistributed(Dense(self.vocab_size, activation='softmax'), name="dec_out")(lstm_out)

        decoder = Model([latent_input, mood_input], output, name='decoder')
        return decoder

    def build_vae(self):
        """Build complete VAE model"""
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        text_input = Input(shape=(self.max_length,), name="vae_text_in")
        mood_input = Input(shape=(1,), name="vae_mood_in")

        z_mean, z_log_var, z = self.encoder([text_input, mood_input])
        reconstruction = self.decoder([z, mood_input])

        self.vae = Model([text_input, mood_input], reconstruction, name='vae')

        # Compile with reconstruction loss; KL is added via add_loss in KLDivergenceLayer
        self.vae.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())

        return self.vae

    # ---------------------------
    # Train / Save / Load
    # ---------------------------
    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """Train the VAE model"""
        print("Loading motivational tips data...")
        tips, moods = self.load_data()

        print("Preprocessing data...")
        X_tips, X_moods = self.preprocess_data(tips, moods)

        print("Building VAE model...")
        self.build_vae()

        print("Training VAE model...")
        # Targets are token ids per timestep (same as inputs for reconstruction)
        X_targets = np.expand_dims(X_tips, -1)  # shape (N, T, 1) for sparse categorical

        history = self.vae.fit(
            [X_tips, X_moods], X_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        self.save_model('models/saved_models/vae_tip_generator.pkl')
        print("VAE training completed!")
        return history

    def save_model(self, filepath):
        """Save VAE model and components"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        encoder_path = filepath.replace('.pkl', '_encoder.keras')
        decoder_path = filepath.replace('.pkl', '_decoder.keras')
        vae_path = filepath.replace('.pkl', '_vae.keras')

        if self.encoder is None or self.decoder is None or self.vae is None:
            raise RuntimeError("Models not built; cannot save.")
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)
        self.vae.save(vae_path)

        components = {
            'tokenizer': self.tokenizer,
            'mood_embeddings': self.mood_embeddings,
            'latent_dim': self.latent_dim,
            'max_length': self.max_length,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(components, f)

        print(f"VAE model saved to {filepath}")

    def load_model(self, filepath):
        """Load VAE model and components"""
        print("[VAE] load_model called with:", filepath)

        encoder_path_keras = filepath.replace('.pkl', '_encoder.keras')
        decoder_path_keras = filepath.replace('.pkl', '_decoder.keras')
        vae_path_keras = filepath.replace('.pkl', '_vae.keras')

        if not all(os.path.exists(p) for p in [encoder_path_keras, decoder_path_keras, vae_path_keras]):
            # Try legacy .h5 fallback
            encoder_path_h5 = filepath.replace('.pkl', '_encoder.h5')
            decoder_path_h5 = filepath.replace('.pkl', '_decoder.h5')
            vae_path_h5 = filepath.replace('.pkl', '_vae.h5')
            if not all(os.path.exists(p) for p in [encoder_path_h5, decoder_path_h5, vae_path_h5]):
                raise FileNotFoundError("VAE model files not found (.keras or .h5). Re-train to generate them.")
            enc_path, dec_path, v_path = encoder_path_h5, decoder_path_h5, vae_path_h5
        else:
            enc_path, dec_path, v_path = encoder_path_keras, decoder_path_keras, vae_path_keras

        try:
            self.encoder = keras_load_model(enc_path, custom_objects={'Sampling': Sampling, 'KLDivergenceLayer': KLDivergenceLayer})
            self.decoder = keras_load_model(dec_path, custom_objects={'Sampling': Sampling, 'KLDivergenceLayer': KLDivergenceLayer})
            self.vae = keras_load_model(v_path, custom_objects={'Sampling': Sampling, 'KLDivergenceLayer': KLDivergenceLayer})
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE models: {e}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"VAE components pickle not found: {filepath}")
        if os.path.getsize(filepath) == 0:
            raise ValueError(f"VAE components pickle is empty: {filepath}. Re-train to regenerate.")
        try:
            with open(filepath, 'rb') as f:
                components = pickle.load(f)
        except EOFError:
            raise ValueError(f"VAE components pickle is corrupted: {filepath}. Re-train to regenerate.")

        self.tokenizer = components.get('tokenizer')
        self.mood_embeddings = components.get('mood_embeddings')
        self.latent_dim = components.get('latent_dim', self.latent_dim)
        self.max_length = components.get('max_length', self.max_length)
        self.vocab_size = components.get('vocab_size', self.vocab_size)

        if self.tokenizer is None or self.mood_embeddings is None:
            raise RuntimeError("Tokenizer or mood_embeddings missing in VAE components pickle.")

        print(f"VAE model loaded from {filepath}")

    # ---------------------------
    # Inference
    # ---------------------------
    def generate_tips(self, mood, context="", num_tips=3, temperature=1.0):
        """Generate motivational tips using VAE"""
        if self.mood_embeddings is None:
            raise RuntimeError("VAE not loaded: missing mood_embeddings.")

        if mood not in self.mood_embeddings['mood_to_idx']:
            mood = 'neutral'
        mood_idx = self.mood_embeddings['mood_to_idx'][mood]

        generated_tips = []
        attempts = 0
        max_attempts = num_tips * 5

        while len(generated_tips) < num_tips and attempts < max_attempts:
            attempts += 1
            z_sample = np.random.normal(0, temperature, (1, self.latent_dim)).astype(np.float32)
            mood_input = np.array([[mood_idx]], dtype=np.int32)

            seq_probs = self.decoder.predict([z_sample, mood_input], verbose=0)  # (1, T, vocab)
            tip_text = self._sequence_to_text(seq_probs[0])
            if self._is_quality_tip(tip_text) and tip_text not in self.generated_tips_cache:
                generated_tips.append(tip_text)
                self.generated_tips_cache.add(tip_text)

        if len(generated_tips) < num_tips:
            generated_tips.extend(self._get_fallback_tips(mood, num_tips - len(generated_tips)))

        return generated_tips[:num_tips]

    def _sequence_to_text(self, sequence_probs):
        """Convert generated sequence probabilities back to text"""
        tokens = np.argmax(sequence_probs, axis=-1)
        words = []
        if self.tokenizer is None:
            return ""
        inv_index = getattr(self.tokenizer, "index_word", {})
        for token in tokens:
            if token > 0 and token in inv_index:
                word = inv_index[token]
                if word:
                    words.append(word)
        text = ' '.join(words).strip()
        if text:
            text = text.upper() + text[1:] if len(text) > 1 else text.upper()
            if not text.endswith('.'):
                text += '.'
        return text

    def _is_quality_tip(self, tip):
        """Simple quality checks"""
        if not tip or len(tip.split()) < 5:
            return False
        if tip.count('_') > 2 or '<' in tip or '>' in tip:
            return False
        meaningful = ['you', 'your', 'yourself', 'take', 'try', 'remember', 'focus', 'breathe']
        if not any(w in tip.lower() for w in meaningful):
            return False
        return True

    def _get_fallback_tips(self, mood, num_tips):
        """Get rule-based fallback tips"""
        fallback_tips_db = {
            'anxious': [
                "Take slow, deep breaths to calm your mind.",
                "Focus on what you can control in this moment.",
                "This anxious feeling is temporary and will pass.",
                "Ground yourself by naming 5 things you can see around you."
            ],
            'happy': [
                "Savor this joyful moment and let it fill your heart.",
                "Share your happiness with someone special today.",
                "Take a moment to appreciate all the good in your life.",
                "Let this positive energy inspire you to help others."
            ],
            'sad': [
                "Be gentle and compassionate with yourself right now.",
                "It's okay to feel sad - your emotions are valid.",
                "Reach out to someone who cares about you.",
                "This difficult time will pass, like all storms do."
            ],
            'tired': [
                "Rest is not a luxury, it's a necessity for your wellbeing.",
                "Listen to your body and give it the rest it needs.",
                "Take a short break to recharge your energy.",
                "Prioritize sleep - it's an investment in your health."
            ],
            'hopeful': [
                "Hold onto this hope and let it guide your actions.",
                "Your optimism is a powerful force for positive change.",
                "Channel this hopeful energy into working toward your goals.",
                "Share your hope with others who need encouragement."
            ],
            'neutral': [
                "Take time to check in with yourself and your needs.",
                "Practice mindfulness and stay present in this moment.",
                "Set one small, meaningful intention for today.",
                "Balance is something you create, not something you find."
            ],
        }
        tips = fallback_tips_db.get(mood, fallback_tips_db['neutral'])
        return random.sample(tips, min(num_tips, len(tips)))

    def retrain(self):
        """Retrain VAE model with new data"""
        print("Retraining VAE model with new data...")
        return self.train()
if __name__ == "__main__":
    vae_generator = VAEMotivationalTipGenerator()
    print("Training VAE Motivational Tip Generator...")
    # Train; adjust epochs to something small first to test end-to-end
    vae_generator.train(epochs=5, batch_size=32, validation_split=0.1)

    # # Quick smoke test: generate a couple of tips after training
    # try:
    #     for mood in ["anxious", "happy", "sad", "tired", "hopeful", "neutral"]:
    #         tips = vae_generator.generate_tips(mood, num_tips=2)
    #         print(f"\n{mood.upper()} tips:")
    #         for i, t in enumerate(tips, 1):
    #             print(f"{i}. {t}")
    # except Exception as e:
    #     print("Generation test failed:", e)