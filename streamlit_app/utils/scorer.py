import pickle


def rule_quality_label(word_count: int, readability: float) -> str:
    """
    Rule-based quality label (must match the notebook logic):

    - High: long & comfortably readable
    - Low: very short or very hard to read
    - Medium: everything else
    """
    if (word_count > 1500) and (50 <= readability <= 70):
        return "High"
    if (word_count < 500) or (readability < 30):
        return "Low"
    return "Medium"


def load_quality_model(path):
    """Load the trained RandomForest model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
