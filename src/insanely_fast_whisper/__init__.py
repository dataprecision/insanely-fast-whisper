"""Convenience helpers for programmatic use of Insanely Fast Whisper."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional

import torch
from transformers import pipeline

from .utils.diarization_pipeline import diarize
from .utils.result import build_result


class WhisperTranscriber:
    """Load the Whisper model once and transcribe multiple files."""

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device_id: str = "0",
        batch_size: int = 24,
        flash: bool = False,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
    ) -> None:
        self.device_id = device_id
        self.batch_size = batch_size
        self.diarization_model = diarization_model
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16,
            device="mps" if device_id == "mps" else f"cuda:{device_id}",
            model_kwargs={"attn_implementation": "flash_attention_2"}
            if flash
            else {"attn_implementation": "sdpa"},
        )

    def transcribe(
        self,
        file_url_or_path: str,
        output_path: str,
        hf_token: str = "no_token",
        *,
        task: str = "transcribe",
        language: Optional[str] = None,
        timestamp: str = "chunk",
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> dict:
        """Transcribe ``file_url_or_path`` and save the JSON result to ``output_path``."""

        ts = "word" if timestamp == "word" else True
        generate_kwargs = {"task": task, "language": language}

        if self.pipe.model.config.name_or_path.endswith(".en"):
            generate_kwargs.pop("task", None)

        outputs = self.pipe(
            file_url_or_path,
            chunk_length_s=30,
            batch_size=self.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=ts,
        )

        if hf_token != "no_token":
            args = SimpleNamespace(
                file_name=file_url_or_path,
                hf_token=hf_token,
                diarization_model=self.diarization_model,
                device_id=self.device_id,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            speakers_transcript = diarize(args, outputs)
            result = build_result(speakers_transcript, outputs)
        else:
            result = build_result([], outputs)

        with open(output_path, "w", encoding="utf8") as fp:
            json.dump(result, fp, ensure_ascii=False)

        return result


def transcribe_and_save(file_url_or_path: str, output_path: str, hf_token: str) -> None:
    """Compatibility wrapper using :class:`WhisperTranscriber`."""
    transcriber = WhisperTranscriber(flash=True, batch_size=15)
    transcriber.transcribe(
        file_url_or_path,
        output_path,
        hf_token,
        task="translate",
        min_speakers=2,
        max_speakers=5,
    )
