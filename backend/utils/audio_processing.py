
import librosa
import numpy as np
class AudioProcessor:
    def extract_features(self, audio_path, sr=22050, duration=3):
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
