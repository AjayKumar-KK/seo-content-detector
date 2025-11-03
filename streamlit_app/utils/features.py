import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def sentence_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def word_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def count_syllables(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    """Compute Flesch Reading Ease score."""
    words = word_tokenize(text)
    sentences = sentence_tokenize(text)
    if len(words) == 0 or len(sentences) == 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    W = len(words)
    S = len(sentences)
    score = 206.835 - 1.015 * (W / S) - 84.6 * (syllables / W)
    return round(score, 2)


def embed_tfidf(texts, max_features: int = 5000):
    """Fit a TF-IDF vectorizer on texts and return (X, vectorizer)."""
    texts = [t if isinstance(t, str) else "" for t in texts]
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vec.fit_transform(texts)
    return X, vec


def top_keywords(vec: TfidfVectorizer, X_row, top_k: int = 5):
    """Extract top-k keywords from a single TF-IDF row."""
    feature_names = np.array(vec.get_feature_names_out())
    row = X_row.toarray().ravel()
    if row.sum() == 0:
        return []
    idx = row.argsort()[-top_k:][::-1]
    return feature_names[idx].tolist()
