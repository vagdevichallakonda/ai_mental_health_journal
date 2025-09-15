
import re
class TextPreprocessor:
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]','', text)
        text = re.sub(r'\s+',' ', text).strip()
        return text
    def extract_emotional_words(self, text, attention_weights):
        words = text.split()
        attn = attention_weights[0][:len(words)]
        indices = (-attn).argsort()[:3]
        return [words[i] for i in indices]
