import numpy as np
import onnxruntime as ort


class SileroVAD:
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        if frame.shape != (512,) or frame.dtype != np.float32:
            raise ValueError('Silero expects 512 float32 samples')
        x = np.concatenate((self.context, frame))[None, :]
        out, self.state = self.session.run(None, {'input': x, 'state': self.state, 'sr': np.array(16000, dtype=np.int64)})
        self.context = x[0, -64:].copy()
        return float(out.item())


def load_silero(path='models/silero_vad.onnx'):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
