import os
import torch
import gc
from typing import List, Dict, Optional, Any, Union

class Transcriber:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        model_cache_dir: Optional[str] = None,
        compute_type: str = "float16",  # only used for whisperx
    ):
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.compute_type = compute_type

        # 1. Auto-resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 2. Prepare cache dirs & Set ENV variables
        self.whisper_dir, self.hf_dir = self._prepare_cache_dirs(model_cache_dir)

    def _prepare_cache_dirs(self, root_dir: Optional[str]):
        if not root_dir:
            return None, None
        w_dir = os.path.join(root_dir, "whisper")
        h_dir = os.path.join(root_dir, "hugging_face")
        os.makedirs(w_dir, exist_ok=True)
        os.makedirs(h_dir, exist_ok=True)

        # Key: set HF_HOME so whisperx (faster-whisper)
        # will look for models under h_dir/hub/models--Systran--faster-whisper-xxx
        os.environ["HF_HOME"] = h_dir
        os.environ["TRANSFORMERS_CACHE"] = h_dir
        os.environ["TORCH_HOME"] = h_dir
        return w_dir, h_dir

    def _release_vram(self, model: Any, is_external: bool = False):
        if is_external:
            return
        if model is not None:
            del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def transcribe(
        self,
        audio_path: str,
        mode: str = "whisperx",
        language: Optional[str] = None,
        external_model: Optional[Any] = None,
    ) -> List[Dict]:
        if mode == "whisper":
            return self._run_whisper_only(audio_path, language, external_model)
        elif mode == "whisperx":
            return self._run_whisperx_only(audio_path, language, external_model)
        elif mode == "mixed":
            return self._run_mixed(audio_path, language, external_model)
        else:
            raise ValueError("Mode must be 'whisper', 'whisperx', or 'mixed'")

    def _run_whisper_only(self, audio_path, language, model=None):
        import whisper
        is_external = model is not None
        print(f"Running transcription with OpenAI Whisper ({self.model_name})...")
        
        if not is_external:
            # The original OpenAI Whisper requires `download_root` to look for .pt files
            # in our custom cache directory.
            model = whisper.load_model(self.model_name, device=self.device, download_root=self.whisper_dir)

        result = model.transcribe(audio_path, language=language, verbose=False)
        
        if not is_external:
            self._release_vram(model)
            
        return result["segments"]

    def _run_whisperx_only(self, audio_path, language, model=None):
        import whisperx
        is_external = model is not None
        print(f"Running transcription with WhisperX ({self.model_name})...")
        
        audio = whisperx.load_audio(audio_path)

        if not is_external:
            # Logic here:
            # 1. We set HF_HOME = self.hf_dir
            # 2. Passing "small" maps internally to "Systran/faster-whisper-small"
            # 3. whisperx will look under HF_HOME for the repo snapshots
            # 4. If the downloader already fetched them, whisperx will load offline files
            model = whisperx.load_model(
                self.model_name, 
                self.device, 
                compute_type=self.compute_type
                # Note: do not pass `download_root`; faster-whisper enables flat mode
                # which breaks the hub-style snapshot directory structure.
            )
        
        result = model.transcribe(audio, batch_size=16, language=language)

        if not is_external:
            self._release_vram(model)

        return self._align_with_whisperx(audio, result, result.get("language"))

    def _run_mixed(self, audio_path, language, model=None):
        import whisper
        import whisperx
        is_external = model is not None
        
        print(f"Running mixed transcription: Whisper {self.model_name} + WhisperX alignment...")

        # 1. Recognition phase: use OpenAI Whisper
        if not is_external:
            model = whisper.load_model(self.model_name, device=self.device, download_root=self.whisper_dir)

        result = model.transcribe(audio_path, language=language, verbose=False)
        detected_lang = result.get("language")

        if not is_external:
            self._release_vram(model)

        # 2. Alignment phase
        print("Performing alignment with WhisperX...")
        audio = whisperx.load_audio(audio_path) 
        return self._align_with_whisperx(audio, result, detected_lang)

    def _align_with_whisperx(self, audio, whisper_result, language_code):
        import whisperx
        try:
            # `model_dir=self.hf_dir` instructs whisperx to look for the Wav2Vec2 model
            # in our cache directory
            model_a, metadata = whisperx.load_align_model(
                language_code=language_code, 
                device=self.device, 
                model_dir=self.hf_dir
            )
            result = whisperx.align(
                whisper_result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False
            )
            self._release_vram(model_a) 
            return result["segments"]
        except Exception as e:
            print(f"WhisperX alignment failed: {e}, returning unaligned segments.")
            return whisper_result["segments"]

def transcribe(
    audio_path: str,
    model_name: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
    model_cache_dir: Optional[str] = None,
    compute_type: str = "float16",
    mode: str = "mixed",
    external_model: Optional[Any] = None,
) -> Union[List[Dict], tuple]:
    t = Transcriber(model_name=model_name, device=device, model_cache_dir=model_cache_dir, compute_type=compute_type)
    segments = t.transcribe(audio_path, mode=mode, language=language, external_model=external_model)
    return segments, t.device
