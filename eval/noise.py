"""Caller voices and background-noise (babble) for the synthetic-caller eval.

CLI:
    python -m eval.noise voices   # fetch English voices from Cartesia -> eval/caller_voices.json
    python -m eval.noise build    # build eval/.cache/babble.pcm (60 s, 16 kHz, int16)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import numpy as np

from server.config import Settings
from server.stages.tts.cartesia import CARTESIA_VERSION, synthesize_bytes

AGENT_VOICE = "a0e99841-438c-4a64-b679-ae501e7d6091"
SAMPLE_RATE = 16000
FRAME = 512
VOICES_PATH = Path(__file__).parent / "caller_voices.json"
CACHE_DIR = Path(__file__).parent / ".cache"
BABBLE_PATH = CACHE_DIR / "babble.pcm"
BABBLE_SECONDS = 60
VOICES_URL = "https://api.cartesia.ai/voices"
TTS_MODEL = "sonic-3.6"

CHATTER = [
    "Yeah, we were here last week and the pasta was amazing, you should try it.",
    "Could we get another bottle of the red, and maybe some bread for the table?",
    "I think the reservation was under my sister's name, but I am not completely sure.",
    "Oh, that looks great, what is the special tonight, is it the risotto?",
    "We should split the check, I had the salad and she had the fish.",
    "Honestly the service here is always so friendly, and the music is not too loud.",
]


def load_voices(path: Path = VOICES_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `uv run python -m eval.noise voices` with CARTESIA_API_KEY set"
        )
    return json.loads(path.read_text())


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def pcm16_to_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec /= np.sqrt(f)
    out = np.fft.irfft(spec, n)
    return (out / rms(out)).astype(np.float32)


async def fetch_voices(http: httpx.AsyncClient, api_key: str) -> list[str]:
    resp = await http.get(
        VOICES_URL,
        params={"language": "en", "limit": 100},
        headers={"Authorization": f"Bearer {api_key}", "Cartesia-Version": CARTESIA_VERSION},
        timeout=30,
    )
    resp.raise_for_status()
    ids = sorted({v["id"] for v in resp.json()["data"]})
    return [i for i in ids if i != AGENT_VOICE][:8]


async def build_babble(http: httpx.AsyncClient, api_key: str, voices: list[str]) -> Path:
    rng = np.random.default_rng(7)
    n_total = BABBLE_SECONDS * SAMPLE_RATE
    mix = np.zeros(n_total, dtype=np.float64)
    for i, sentence in enumerate(CHATTER):
        pcm = await synthesize_bytes(
            http, api_key, TTS_MODEL, voices[i % len(voices)], sentence, SAMPLE_RATE
        )
        utt = pcm16_to_f32(pcm)
        utt = utt / max(rms(utt), 1e-9) * 0.1
        pos = int(rng.integers(0, SAMPLE_RATE * 3))
        while pos < n_total:
            seg = utt[: n_total - pos]
            mix[pos : pos + len(seg)] += seg
            pos += len(utt) + int(rng.integers(SAMPLE_RATE // 2, SAMPLE_RATE * 3))
    speech_rms = max(rms(mix), 1e-9)
    mix = mix / speech_rms * 0.1
    mix += pink_noise(n_total, rng) * 10 ** (-30 / 20)
    mix = mix / max(rms(mix), 1e-9) * 0.1
    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix *= 0.99 / peak
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BABBLE_PATH.write_bytes((mix * 32767).astype("<i2").tobytes())
    return BABBLE_PATH


async def ensure_babble(http: httpx.AsyncClient, api_key: str) -> Path:
    """Return the babble path, regenerating it if missing."""
    if not BABBLE_PATH.exists():
        await build_babble(http, api_key, load_voices())
    return BABBLE_PATH


class NoiseSource:
    """Endless babble at a fixed SNR relative to caller speech normalized to -20 dBFS."""

    def __init__(self, babble: np.ndarray, snr_db: float, offset: int = 0) -> None:
        unit = babble / max(rms(babble), 1e-9)
        self._noise = (unit * 10 ** ((-20 - snr_db) / 20)).astype(np.float32)
        self._pos = offset % len(self._noise)

    @classmethod
    def from_file(cls, snr_db: float, offset: int = 0) -> NoiseSource:
        return cls(pcm16_to_f32(BABBLE_PATH.read_bytes()), snr_db, offset)

    def next_frame(self, n: int = FRAME) -> np.ndarray:
        idx = (self._pos + np.arange(n)) % len(self._noise)
        self._pos = (self._pos + n) % len(self._noise)
        return self._noise[idx]


async def _main(cmd: str) -> int:
    settings = Settings()
    if not settings.cartesia_api_key:
        print("CARTESIA_API_KEY is not set", file=sys.stderr)
        return 1
    async with httpx.AsyncClient() as http:
        if cmd == "voices":
            voices = await fetch_voices(http, settings.cartesia_api_key)
            VOICES_PATH.write_text(json.dumps(voices, indent=2) + "\n")
            print(f"wrote {len(voices)} voices to {VOICES_PATH}")
        else:
            path = await build_babble(http, settings.cartesia_api_key, load_voices())
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["voices", "build"])
    sys.exit(asyncio.run(_main(ap.parse_args().command)))
