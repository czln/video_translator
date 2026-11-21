import gc
import os
from typing import Any, Dict, List, Optional, Tuple


class Transcriber:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        model_cache_dir: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.compute_type = compute_type if compute_type is not None else (
            "int8" if device == "cpu" else "float16"
        )
        self.whisper_dir, self.hf_dir = self._prepare_cache_dirs(model_cache_dir)

    def _prepare_cache_dirs(self, root_dir: Optional[str]):
        if not root_dir:
            return None, None
        w_dir = os.path.join(root_dir, "whisper")
        h_dir = os.path.join(root_dir, "hugging_face")
        os.makedirs(w_dir, exist_ok=True)
        os.makedirs(h_dir, exist_ok=True)
        os.environ["HF_HOME"] = h_dir
        os.environ["TRANSFORMERS_CACHE"] = h_dir
        os.environ["TORCH_HOME"] = h_dir
        return w_dir, h_dir

    def _release_vram(self):
        gc.collect()
        if self.device.split(":")[0] == "cuda":
            import torch

            torch.cuda.empty_cache()

    def transcribe(
        self,
        audio_path: str,
        mode: str = "whisperx",
        language: Optional[str] = None,
        external_model: Optional[Any] = None,
    ) -> List[Dict]:
        segments, _, _ = self.transcribe_with_language(audio_path, mode, language, external_model)
        return segments

    def transcribe_with_language(
        self,
        audio_path: str,
        mode: str = "whisperx",
        language: Optional[str] = None,
        external_model: Optional[Any] = None,
    ) -> Tuple[List[Dict], str, Optional[str]]:
        model = external_model
        if mode == "whisperx":
            import whisperx

            print(f"Running transcription with WhisperX ({self.model_name})...")
            audio = whisperx.load_audio(audio_path)
            if model is None:
                device, _, index = self.device.partition(":")
                model = whisperx.load_model(
                    self.model_name,
                    device,
                    device_index=int(index) if index else 0,
                    compute_type=self.compute_type,
                    download_root=self.hf_dir,
                )
            options = {"batch_size": 16}
        elif mode in ("whisper", "mixed"):
            print(f"Running transcription with OpenAI Whisper ({self.model_name})...")
            audio = audio_path
            if model is None:
                import whisper

                model = whisper.load_model(
                    self.model_name, device=self.device, download_root=self.whisper_dir
                )
            options = {"verbose": False}
        else:
            raise ValueError("Mode must be 'whisper', 'whisperx', or 'mixed'")

        try:
            result = model.transcribe(audio, language=language, **options)
        finally:
            if external_model is None:
                del model
                self._release_vram()

        detected_language = result.get("language") or language
        segments = result["segments"]
        if mode == "mixed":
            import whisperx

            audio = whisperx.load_audio(audio_path)
        if mode != "whisper":
            print("Performing alignment with WhisperX...")
            segments = self._align_with_whisperx(audio, segments, detected_language)
        return segments, self.device, detected_language

    def _align_with_whisperx(self, audio, segments, language_code):
        import whisperx

        model = None
        try:
            model, metadata = whisperx.load_align_model(
                language_code=language_code, device=self.device, model_dir=self.hf_dir
            )
            result = whisperx.align(
                segments, model, metadata, audio, self.device, return_char_alignments=False
            )
            return result["segments"]
        except Exception as e:
            print(f"WhisperX alignment failed: {e}, returning unaligned segments.")
            return segments
        finally:
            if model is not None:
                del model
                self._release_vram()


def transcribe_with_language(
    audio_path: str,
    model_name: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
    model_cache_dir: Optional[str] = None,
    compute_type: Optional[str] = None,
    mode: str = "mixed",
    external_model: Optional[Any] = None,
) -> Tuple[List[Dict], str, Optional[str]]:
    """Return subtitle segments, the resolved device, and the recognition language."""
    transcriber = Transcriber(
        model_name=model_name, device=device, model_cache_dir=model_cache_dir, compute_type=compute_type
    )
    return transcriber.transcribe_with_language(audio_path, mode, language, external_model)


def transcribe(
    audio_path: str,
    model_name: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
    model_cache_dir: Optional[str] = None,
    compute_type: Optional[str] = None,
    mode: str = "mixed",
    external_model: Optional[Any] = None,
) -> Tuple[List[Dict], str]:
    segments, used_device, _ = transcribe_with_language(
        audio_path, model_name, language, device, model_cache_dir, compute_type, mode, external_model
    )
    return segments, used_device
