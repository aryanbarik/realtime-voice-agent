from pathlib import Path
import shutil

import httpx
from huggingface_hub import hf_hub_download

SILERO_URL = 'https://raw.githubusercontent.com/snakers4/silero-vad/v6.3/src/silero_vad/data/silero_vad.onnx'
SMART_TURN_MODEL = 'smart-turn-v3.2-cpu.onnx'


def ensure_models(dir='models'):
    target = Path(dir)
    target.mkdir(parents=True, exist_ok=True)
    silero = target / 'silero_vad.onnx'
    if not silero.exists():
        with httpx.stream('GET', SILERO_URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            temp = silero.with_suffix('.tmp')
            try:
                with temp.open('wb') as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                temp.replace(silero)
            finally:
                temp.unlink(missing_ok=True)
    smart = target / SMART_TURN_MODEL
    if not smart.exists():
        source = hf_hub_download('pipecat-ai/smart-turn-v3', SMART_TURN_MODEL)
        temp = smart.with_suffix('.tmp')
        try:
            shutil.copyfile(source, temp)
            temp.replace(smart)
        finally:
            temp.unlink(missing_ok=True)
