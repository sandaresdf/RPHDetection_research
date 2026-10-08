# detector.py
"""
Extracts features from a CoT trace and retrieved docs and trains a simple detector.
Features (example):
 - presence_of_placeholder (binary)
 - cot_length (int)
 - avg_step_len (float)
 - poisoned_doc_fraction (float)
Simple sklearn pipeline is used (LogisticRegression).
"""

import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

PLACEHOLDER_PATTERNS = [
    r"\[POISONED_REASONING_STEP_1\]",
    r"\[BIAS_DIRECT_PREFER_X\]",
    r"\[BIAS_INDIRECT_FRAME\]",
    r"\[BIAS_OBFUSCATED_A\]"
]

def extract_features(cot_steps: list, retrieved: list) -> np.ndarray:
    cot_text = " ".join(cot_steps)
    presence = int(any(re.search(p, cot_text) for p in PLACEHOLDER_PATTERNS))
    cot_length = len(cot_text.split())
    step_lens = [len(s.split()) for s in cot_steps] if cot_steps else [0]
    avg_step_len = float(np.mean(step_lens))
    poisoned_frac = float(sum(1 for r in retrieved if r.get("poisoned_flag")) / max(1, len(retrieved)))
    # feature vector
    return np.array([presence, cot_length, avg_step_len, poisoned_frac], dtype=float).reshape(1, -1)

class SimpleDetector:
    def __init__(self):
        self.model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, feat: np.ndarray) -> float:
        return float(self.model.predict_proba(feat)[0, 1])

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
