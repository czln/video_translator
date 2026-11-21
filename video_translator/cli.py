import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .model_downloader import ModelDownloader
from .transcriber import transcribe_with_language
from .translator import translate_segments
from .srt import write_srt

# Try importing ffmpeg-python, handle case where user hasn't installed it
try:
    import ffmpeg
except ImportError:
    ffmpeg = None


def _source_version(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()), "device": stat.st_dev, "inode": stat.st_ino,
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
    }


def _cache_path(directory, input_path, kind, *key):
    # JSON preserves boundaries between language, model and path components.
    identity = json.dumps([str(input_path.resolve()), *key], ensure_ascii=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return directory / f"{input_path.stem[:40]}.{kind}.{digest}.json"


def _load_cache(path, source):
    try:
        with path.open(encoding="utf-8") as cache:
            data = json.load(cache)
        if data.get("source") == source and isinstance(data.get("segments"), list):
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _save_cache(path, source, **payload):
    # Publish a complete JSON file even if another process is using this cache.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as cache:
            temporary = Path(cache.name)
            json.dump({"source": source, **payload}, cache, ensure_ascii=False)
        os.replace(temporary, path)
    except OSError as exc:
        print(f"Could not save cache {path}: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _audio_for(input_path, cache_dir, source, force):
    cached_audio = None
    if cache_dir is not None:
        key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:24]
        cached_audio = cache_dir / f"{input_path.stem[:40]}.{key}.audio.wav"
        if not force and cached_audio.is_file() and cached_audio.stat().st_size:
            print(f"Reusing cached audio: {cached_audio}")
            yield str(cached_audio)
            return

    with tempfile.TemporaryDirectory(dir=cache_dir) as scratch:
        wav = Path(scratch) / "audio.wav"
        print("Extracting audio...")
        extract_audio(str(input_path), str(wav))
        if cached_audio is not None:
            os.replace(wav, cached_audio)
            wav = cached_audio
        yield str(wav)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device string."""
    if device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'
    return device


def extract_audio(video_path: str, out_audio_path: str) -> None:
    """Extract mono 16k WAV using ffmpeg-python."""
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is not installed. Run: pip install ffmpeg-python")

    try:
        # Construct the ffmpeg command graph
        # equivalent to: ffmpeg -y -i input -ar 16000 -ac 1 -vn output.wav
        (
            ffmpeg
            .input(video_path)
            .output(out_audio_path, ar=16000, ac=1, vn=None)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg Error!")
        if e.stderr:
            print(e.stderr.decode('utf8'))
        raise RuntimeError("Failed to extract audio using FFmpeg.") from e


def main():
    parser = argparse.ArgumentParser(prog="video-translator")
    parser.add_argument("input", nargs="?", help="Input video file")
    parser.add_argument("--model", "-m", default="small", help="Whisper model name (tiny, base, small, medium, large)")
    parser.add_argument("--transcriber", choices=["whisper", "whisperx", "mixed"], default="mixed", help="Transcriber backend to use: 'whisper', 'whisperx', or 'mixed' (default 'mixed')")
    parser.add_argument("--lang", "-l", default=None, help="Target language code (e.g., fr, es). If omitted, only transcribe.")
    parser.add_argument("--src-lang", default=None, help="Source/speech language code (e.g., en, fr). If omitted Whisper will attempt to detect language.")
    
    # Modified: removed choices restriction to allow 'model:...' syntax
    parser.add_argument("--translator-mode", default="online", 
                        help="Translation backend: 'online' (google), 'offline' (argos), 'm2m100' (default 418M), or 'model:<repo_id>' (e.g. model:facebook/m2m100_1.2B).")
    
    parser.add_argument("--proxy", default=None, help="Proxy URL to use for online translator (e.g. http://127.0.0.1:8080).")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-segment translation timeout in seconds (default 10.0).")
    parser.add_argument("--device", default="auto", help="Device to run Whisper on: 'auto' (default), 'cpu', or 'cuda[:<idx>]' (e.g. 'cuda:0').")
    parser.add_argument("--cache-dir", default=None, help="Directory to store transcript/translation cache files. If omitted caches are stored next to the source video.")
    parser.add_argument("--model-cache-dir", default=None, help="Directory where Hugging Face models will be downloaded/cached. Supports resuming downloads where supported.")
    parser.add_argument("--hf-mirror", default=None, help="Hugging Face mirror endpoint to use (sets HF_ENDPOINT). e.g. https://hf-mirror.example.com")
    parser.add_argument("--download-models", action='store_true', help="Prefetch common Whisper and Hugging Face models to the model cache and exit.")
    parser.add_argument("--auto-install-argos", action='store_true', help="When using offline translator, automatically download and install Argos Translate language package if available.")
    parser.add_argument("--output", "-o", default=None, help="Output SRT path")
    parser.add_argument("--force", action='store_true', help="Force re-transcribe and re-translate, ignoring caches")
    args = parser.parse_args()
    
    if args.hf_mirror:
        args.hf_mirror = args.hf_mirror.rstrip("/")
        os.environ["HF_ENDPOINT"] = args.hf_mirror

    if args.download_models:
        downloader = ModelDownloader(proxy=args.proxy, hf_mirror=args.hf_mirror)
        root = os.path.abspath(args.model_cache_dir or "./model-cache")
        try:
            results = downloader.download_all_models(save_root=root, model_size=args.model)
        except RuntimeError as exc:
            parser.exit(1, f"{exc}\n")
        for result in results:
            print(" -", result)
        return

    if not args.input:
        parser.error("input is required unless --download-models is set")
    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    suffix = f".{args.lang}.srt" if args.lang else ".srt"
    out_srt = Path(args.output) if args.output else input_path.with_suffix(suffix)
    if out_srt.exists() and input_path.samefile(out_srt):
        parser.error("Output SRT must be a different file from the input video")

    source = _source_version(input_path)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cache_parent = cache_dir or input_path.parent
    args.device = _resolve_device(args.device)
    asr_key = (args.model, args.transcriber, args.src_lang)
    transcript_path = _cache_path(cache_parent, input_path, "transcript", *asr_key)
    transcript = None if args.force else _load_cache(transcript_path, source)
    if transcript is None:
        with _audio_for(input_path, cache_dir, source, args.force) as audio:
            print("Transcribing audio...")
            segments, device, detected = transcribe_with_language(
                audio, model_name=args.model, language=args.src_lang, device=args.device,
                model_cache_dir=args.model_cache_dir, mode=args.transcriber,
            )
        print(f"Using device: {device}")
        _save_cache(transcript_path, source, segments=segments, language=detected)
    else:
        print(f"Loaded transcript from cache: {transcript_path}")
        segments = transcript["segments"]
        detected = transcript.get("language")

    if args.lang:
        language = args.src_lang
        if language is None and args.translator_mode.lower() not in ("online", "google"):
            language = detected
        translation_path = _cache_path(cache_parent, input_path, "translated", *asr_key,
                                       language, args.translator_mode, args.lang)
        # A forced transcription can change the wording or timing for the same source.
        cues = [(seg.get("start"), seg.get("end"), seg.get("text")) for seg in segments]
        transcript_digest = hashlib.sha256(json.dumps(cues).encode()).hexdigest()
        translation = None if args.force else _load_cache(translation_path, source)
        if translation is None or translation.get("transcript_digest") != transcript_digest:
            print(f"Translating to {args.lang}...")
            try:
                segments = translate_segments(
                    segments, args.lang, mode=args.translator_mode, proxy=args.proxy,
                    timeout=args.timeout, src_lang=language, auto_install=args.auto_install_argos,
                    device=args.device, model_cache_dir=args.model_cache_dir,
                )
            except RuntimeError as exc:
                parser.exit(1, f"Error during translation: {exc}\n")
            _save_cache(translation_path, source, segments=segments, transcript_digest=transcript_digest)
        else:
            print(f"Loaded translation from cache: {translation_path}")
            segments = translation["segments"]

    write_srt(segments, out_srt, use_translated=bool(args.lang))
    print(f"Done. Output: {out_srt}")
