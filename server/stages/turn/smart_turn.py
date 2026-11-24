from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path("models/smart-turn-v3.2-cpu.onnx")
MAX_SAMPLES = 8 * 16000

_lock = threading.Lock()
_session: ort.InferenceSession | None = None
_extractor = None


def _load(path: Path = MODEL_PATH):
    global _session, _extractor
    with _lock:
        if _session is None:
            from transformers import WhisperFeatureExtractor

            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _extractor = WhisperFeatureExtractor(chunk_length=8)
            _session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
    return _session, _extractor


def predict(audio: np.ndarray) -> float:
    """Probability that the turn is complete. `audio` is float32 16 kHz mono."""
    session, extractor = _load()
    audio = audio[-MAX_SAMPLES:]
    feats = extractor(
        audio,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_SAMPLES,
        truncation=True,
        do_normalize=True,
    ).input_features.astype(np.float32)
    outputs = session.run(None, {"input_features": feats})
    return float(outputs[0][0].item())


class SmartTurnDetector:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    async def predict(self, audio: np.ndarray) -> float:
        return await asyncio.to_thread(predict, audio)

    async def is_complete(self, audio: np.ndarray) -> bool:
        return await self.predict(audio) >= self.threshold
