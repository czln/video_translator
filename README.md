# video-translator

Command-line tool to extract audio from a video, transcribe speech, translate text, and produce subtitles (SRT).

Requirements
- `ffmpeg` installed on your system (used to extract audio).
- Python packages from `requirements.txt` (see below).


Install

```bash
python -m pip install -r requirements.txt
# WhisperX requires `torch` and `whisper` under the hood. Install `torch` separately
# with an appropriate wheel for your platform. Example (CPU-only):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Quick usage

```bash
# transcribe only (no translation)
python -m video_translator path/to/video.mp4 --model small

# transcribe and translate to French
python -m video_translator path/to/video.mp4 --model small --lang fr

# specify output SRT file
python -m video_translator path/to/video.mp4 --lang es -o subs.es.srt
```

Notes
- The tool uses `whisper` for speech-to-text and `whisperx` to align timestamps (better SRT timing).
- Choose `--model tiny|base|small|medium|large` according to available RAM/CPU/GPU.
- Translations support multiple modes: `online` (Google via `deep-translator`), `offline` (via `argostranslate`), and `m2m100` models. See `--translator-mode` for details.

Files
- `video_translator/cli.py` — CLI entrypoint and audio extraction
	- `--src-lang`: supply source speech language code (e.g. `en`) to improve transcription accuracy.
	- `--transcriber`: choose the transcription backend: `whisper`, `whisperx`, or `mixed` (default `mixed`).
	- `--translator-mode`: choose translation backend or specific model. Examples:
	  - `online` — uses `deep-translator`'s `GoogleTranslator`.
	  - `offline` — uses `argostranslate` (requires installed packages).
	  - `m2m100` — use the default `facebook/m2m100_418M` offline model.
	  - `m2m100_1.2B`, `m2m100_418M` — select a specific M2M100 variant (maps to `facebook/m2m100_<variant>`).
	  - `model:facebook/m2m100_1.2B` — pass an explicit Hugging Face repo id to use as the model.
	- `--proxy`: when using `--translator-mode online`, set an HTTP(S) proxy (e.g. `http://127.0.0.1:8080`).
	- `--timeout`: per-segment translation timeout in seconds (default `10.0`).
	- `--device`: select device for Whisper: `auto` (default), `cpu`, or `cuda[:<index>]` (e.g. `cuda:0`).
	- `--cache-dir`: optional directory to store transcript/translation cache files. If omitted caches are stored beside the source video file.
	- `--model-cache-dir`: optional directory where Hugging Face/Whisper models will be downloaded and cached. When provided, the CLI exports `HF_HOME`, `TRANSFORMERS_CACHE`, and `TORCH_HOME` to point into this directory so downstream loaders reuse the cache.
	- `--hf-mirror`: optional Hugging Face mirror endpoint (sets `HF_ENDPOINT`) to speed up model downloads behind mirrors.
	- `--auto-install-argos`: when using `--translator-mode offline`, automatically attempt to download and install an Argos Translate language package for the requested language pair. If not set and no package is installed, the program will exit with an error.
- `video_translator/transcriber.py` — wrapper for Whisper transcription
- `video_translator/translator.py` — wrapper around `deep-translator` for segment-wise translation
- `video_translator/srt.py` — write segments to SRT
