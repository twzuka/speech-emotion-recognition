"""Проверка контрольных сумм модели и одного вычисления на CPU."""

import hashlib
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from src.predict import EmotionPredictor

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "models" / "20261006-190944-469997"
HASHES = {
    "best.keras": "2ee1466f1abc2d55e0b277da583d7699315973c3ed77f4bf38ec4d725ba01fd0",
    "run_info.json": "0114dea7f1b6205f7942e718a6a707648a504fd52f23b44f0d9487c709a3ba4b",
}


def main() -> None:
    for name, expected in HASHES.items():
        actual = hashlib.sha256((RUN_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}. Extract the release again.")

    predictor = EmotionPredictor(RUN_DIR)
    features = np.zeros((1, *predictor.model.input_shape[1:]), dtype=np.float32)
    with tf.device("/CPU:0"):
        output = predictor.model(features, training=False).numpy()
    if (
        output.shape != (1, 8)
        or not np.isfinite(output).all()
        or np.any(output < 0)
        or np.any(output > 1)
        or not np.allclose(output.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("The CPU inference returned invalid probabilities")
    print("Model checksums, settings, shapes and CPU inference: OK")
    print(f"TensorFlow {tf.__version__}; parameters: {predictor.model.count_params()}")


if __name__ == "__main__":
    main()
