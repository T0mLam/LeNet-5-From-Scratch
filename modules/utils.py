from __future__ import annotations

import pickle


def save_model(model: Sequential, filename: str) -> bool:
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return True


def load_model(filename: str) -> Sequential:
    with open(filename, 'rb') as f:
        pre_trained_model = pickle.load(f)
    return pre_trained_model