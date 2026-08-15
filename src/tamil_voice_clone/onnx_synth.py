from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from .config import RuntimeConfig
from .model import SpeakerCondition
from .tokenizer import TokenizedPhonemes


class OnnxSynthesizerError(RuntimeError):
    pass


class OnnxSynthesizer:
    """CPU-only ONNX synthesizer for the compact bilingual model we will train.

    The exported model contract is intentionally small and stable:

    inputs
      token_ids: int64 [1, T]
      language_ids: int64 [1, T]
      speaker_embedding: float32 [1, D]

    output
      waveform: float32 [1, N] or [N]

    Keeping text/G2P and speaker extraction outside the acoustic model makes
    quantization and memory profiling easier and avoids shipping PyTorch.
    """

    INPUT_NAMES = ("token_ids", "language_ids", "speaker_embedding")

    def __init__(
        self,
        model_path: Path,
        sample_rate: int = 22050,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        if not model_path.is_file():
            raise OnnxSynthesizerError(f"Synthesizer model not found: {model_path}")

        self.sample_rate = sample_rate
        self.runtime_config = runtime_config or RuntimeConfig()

        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise OnnxSynthesizerError(
                "onnxruntime is not installed; install the 'onnx' extra."
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.runtime_config.intra_op_threads
        options.inter_op_num_threads = self.runtime_config.inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        available_inputs = {item.name for item in self._session.get_inputs()}
        missing = set(self.INPUT_NAMES) - available_inputs
        if missing:
            raise OnnxSynthesizerError(
                "Synthesizer ONNX model is missing required inputs: "
                + ", ".join(sorted(missing))
            )

    def synthesize_tokens(
        self,
        tokens: TokenizedPhonemes,
        speaker: SpeakerCondition,
    ) -> np.ndarray:
        token_ids = np.asarray(tokens.token_ids, dtype=np.int64)[None, :]
        language_ids = np.asarray(tokens.language_ids, dtype=np.int64)[None, :]
        speaker_embedding = np.asarray(speaker.embedding, dtype=np.float32).reshape(1, -1)

        if token_ids.shape != language_ids.shape:
            raise OnnxSynthesizerError("Token and language input shapes do not match.")
        if speaker_embedding.shape[1] == 0:
            raise OnnxSynthesizerError("Speaker embedding is empty.")

        outputs = self._session.run(
            None,
            {
                "token_ids": token_ids,
                "language_ids": language_ids,
                "speaker_embedding": speaker_embedding,
            },
        )
        if not outputs:
            raise OnnxSynthesizerError("Synthesizer returned no outputs.")

        waveform = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if waveform.size == 0 or not np.all(np.isfinite(waveform)):
            raise OnnxSynthesizerError("Synthesizer returned invalid audio.")
        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak
        return waveform
