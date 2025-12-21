import argparse
import json
import os
import tempfile
from typing import Optional, List, Dict
from .model_downloader import ModelDownloader

from .transcriber import transcribe
from .translator import translate_segments
from .srt import write_srt

# Try importing ffmpeg-python, handle case where user hasn't installed it
try:
    import ffmpeg
except ImportError:
    ffmpeg = None


def _cache_paths_for(input_path: str, model: str, target_lang: Optional[str] = None, src_lang: Optional[str] = None, translator_mode: Optional[str] = None, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Return cache file paths. If `cache_dir` is provided, store caches there using the video's base name.

    Otherwise default to using an absolute path next to the original file (legacy behavior).
    """
    src_tag = f".{src_lang}" if src_lang else ""
    # Sanitize translator mode for filename (replace : or / with _)
    mode_str = str(translator_mode).replace(':', '_').replace('/', '_') if translator_mode else ""
    mode_tag = f".{mode_str}" if mode_str else ""

    if cache_dir:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        transcript_cache = os.path.join(cache_dir, f"{base_name}.transcript{src_tag}.{model}.json")
        translation_cache = None
        if target_lang:
            translation_cache = os.path.join(cache_dir, f"{base_name}.translated{src_tag}{mode_tag}.{target_lang}.{model}.json")
    else:
        base = os.path.abspath(input_path)
        transcript_cache = f"{base}.transcript{src_tag}.{model}.json"
        translation_cache = None
        if target_lang:
            translation_cache = f"{base}.translated{src_tag}{mode_tag}.{target_lang}.{model}.json"

    return {"transcript": transcript_cache, "translation": translation_cache}


def _load_cache(path: str, input_path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # validate source mtime
        src_mtime = os.path.getmtime(input_path)
        if data.get('source_mtime') != src_mtime:
            return None
        return data
    except Exception:
        return None


def _save_cache(path: str, input_path: str, payload: dict) -> None:
    try:
        # ensure parent directory exists
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload_to_save = dict(payload)
        payload_to_save['source_mtime'] = os.path.getmtime(input_path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload_to_save, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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
    parser.add_argument("input", help="Input video file")
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
    
    # -------------------------------------------------------------------------
    # FIX: Apply HF Mirror environment variable immediately after parsing args
    # This ensures transformers/whisperx use the mirror during execution.
    # -------------------------------------------------------------------------
    if args.hf_mirror:
        args.hf_mirror = args.hf_mirror.rstrip('/')
        os.environ["HF_ENDPOINT"] = args.hf_mirror
        print(f"Hugging Face mirror set as: {args.hf_mirror}")

    # Resolve device for M2M100 (convert 'auto' to actual device)
    args.device = _resolve_device(args.device)

    input_path = args.input
    out_srt = args.output or os.path.splitext(input_path)[0] + (f'.{args.lang}.srt' if args.lang else '.srt')
    # normalize/cache-dir: make absolute path and create if necessary
    if args.cache_dir:
        args.cache_dir = os.path.abspath(args.cache_dir)
        try:
            os.makedirs(args.cache_dir, exist_ok=True)
        except Exception:
            pass

    # normalize model-cache-dir if provided
    if args.model_cache_dir:
        args.model_cache_dir = os.path.abspath(args.model_cache_dir)
        try:
            os.makedirs(args.model_cache_dir, exist_ok=True)
        except Exception:
            pass
        # create subdirectories for clarity and print them
        whisper_dir = os.path.join(args.model_cache_dir, 'whisper')
        hf_dir = os.path.join(args.model_cache_dir, 'hugging_face')
        try:
            os.makedirs(whisper_dir, exist_ok=True)
        except Exception:
            pass
        try:
            os.makedirs(hf_dir, exist_ok=True)
        except Exception:
            pass
        print(f"Model cache (root): {args.model_cache_dir}")
        print(f" - Whisper cache: {whisper_dir}")
        print(f" - Hugging Face cache: {hf_dir}")

    # if user requested to prefetch models, do that and exit early
    if args.download_models:
        downloader = ModelDownloader(proxy=args.proxy, hf_mirror=args.hf_mirror)
        root = args.model_cache_dir or os.path.abspath('./model-cache')
        os.makedirs(root, exist_ok=True)
        print(f"Prefetching models into: {root} (this may take a while)")
        results = downloader.download_all_models(save_root=root, include_whisper=True, include_hf=True)
        print("Prefetch results:")
        for r in results:
            print(" -", r)
        print("Done. Exiting as --download-models was specified.")
        return

    # prepare cache paths (include source language in cache name)
    # include transcriber backend in cache key so different backends produce separate caches
    model_cache_tag = f"{args.model}.{args.transcriber}"
    caches = _cache_paths_for(input_path, model_cache_tag, args.lang, src_lang=args.src_lang, translator_mode=args.translator_mode, cache_dir=args.cache_dir)

    if args.cache_dir:
        print(f"Using cache directory: {args.cache_dir}")

    segments = None

    # try loading transcript cache
    if not args.force:
        tcache = caches.get('transcript')
        if tcache and os.path.exists(tcache):
            data = _load_cache(tcache, input_path)
            if data and 'segments' in data:
                print('Loaded transcript from cache:', tcache)
                segments = data['segments']
                # normalize segments: ensure each is a dict and has string 'text'
                normalized = []
                for seg in segments:
                    if not isinstance(seg, dict):
                        seg = {'start': 0.0, 'end': 0.0, 'text': str(seg or '')}
                    # ensure text is string
                    txt = seg.get('text')
                    if txt is None:
                        seg['text'] = ''
                    else:
                        seg['text'] = str(txt)
                    normalized.append(seg)
                segments = normalized

    # Determine audio cache path: prefer storing audio in `--cache-dir` so ffmpeg
    # extraction can be skipped on subsequent runs for the same video.
    created_temp_audio = False
    if args.cache_dir:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        audio_path = os.path.join(args.cache_dir, f"{base_name}.audio.wav")
    else:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_path = tmp.name
        created_temp_audio = True

    try:
        if segments is None:
            # If audio already exists in cache, skip extraction.
            if os.path.exists(audio_path):
                print(f"Reusing cached audio: {audio_path}")
            else:
                print('Extracting audio...')
                extract_audio(input_path, audio_path)

            print('Transcribing audio (this may take a while)...')
            segments, used_device = transcribe(
                audio_path,
                model_name=args.model,
                language=args.src_lang,
                device=args.device,
                model_cache_dir=args.model_cache_dir,
                mode=args.transcriber,
            )
            print(f"Using device: {used_device}")

            # save transcript cache
            tcache = caches.get('transcript')
            if tcache:
                _save_cache(tcache, input_path, {'segments': segments, 'model': args.model, 'device': args.device})

        if args.lang:
            # try loading translation cache
            translated = None
            if not args.force:
                trcache = caches.get('translation')
                if trcache and os.path.exists(trcache):
                    data = _load_cache(trcache, input_path)
                    if data and 'segments' in data and data.get('target_lang') == args.lang:
                        print('Loaded translation from cache:', trcache)
                        translated = data['segments']
                        # normalize translated segments to ensure 'translated_text' is string
                        normalized = []
                        for seg in translated:
                            if not isinstance(seg, dict):
                                seg = {'start': 0.0, 'end': 0.0, 'translated_text': str(seg or '')}
                            ttxt = seg.get('translated_text')
                            if ttxt is None:
                                seg['translated_text'] = ''
                            else:
                                seg['translated_text'] = str(ttxt)
                            # also ensure 'text' exists as string for fallback
                            txt = seg.get('text')
                            if txt is None:
                                seg['text'] = ''
                            else:
                                seg['text'] = str(txt)
                            normalized.append(seg)
                        translated = normalized

            if translated is None:
                print(f'Translating to {args.lang}...')
                try:
                    translated = translate_segments(
                        segments,
                        args.lang,
                        mode=args.translator_mode,
                        proxy=args.proxy,
                        timeout=args.timeout,
                        src_lang=args.src_lang,
                        auto_install=args.auto_install_argos,
                        device=args.device,
                        model_cache_dir=args.model_cache_dir,
                    )
                except RuntimeError as e:
                    print('Error during translation:', e)
                    print('Aborting.')
                    raise SystemExit(1)
                trcache = caches.get('translation')
                if trcache:
                    _save_cache(trcache, input_path, {'segments': translated, 'target_lang': args.lang, 'model': args.model, 'translator_mode': args.translator_mode, 'device': args.device})

            write_srt(translated, out_srt, use_translated=True)
        else:
            write_srt(segments, out_srt, use_translated=False)

        print('Done. Output:', out_srt)

    finally:
        # Only remove the audio file if it was a temporary file; preserve cached audio.
        try:
            if created_temp_audio and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
