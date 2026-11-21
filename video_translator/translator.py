import concurrent.futures
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


def _normalize_lang_code(code: Optional[str]) -> Optional[str]:
    """Helper to normalize language codes (e.g. 'zh-CN' -> 'zh')."""
    if not code:
        return None
    c = str(code).strip().lower()
    return c.split('-')[0]

# --- Translator Classes ---

class BaseTranslator(ABC):
    def __init__(self, src_lang: Optional[str], target_lang: str):
        self.src_lang = src_lang
        self.target_lang = target_lang

    @abstractmethod
    def translate(self, text: str) -> str:
        pass


class DeepGoogleTranslator(BaseTranslator):
    def __init__(self, src_lang: Optional[str], target_lang: str, proxy: Optional[str] = None):
        super().__init__(src_lang, target_lang)
        self.proxy = proxy
        self.src = src_lang if src_lang else 'auto'

    def translate(self, text: str) -> str:
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            raise RuntimeError("deep-translator not installed. Run: pip install deep-translator")
        
        t = GoogleTranslator(source=self.src, target=self.target_lang)
        return t.translate(text)


class ArgoTranslator(BaseTranslator):
    def __init__(self, src_lang: Optional[str], target_lang: str, auto_install: bool = False):
        super().__init__(src_lang, target_lang)
        
        if not self.src_lang:
            raise RuntimeError("ArgoTranslator requires 'src_lang' to be specified.")

        try:
            import argostranslate.package as argopackage
            import argostranslate.translate as argotrans
        except ImportError:
            raise RuntimeError("argostranslate not installed. Run: pip install argostranslate")

        self.argopackage = argopackage
        self.argotrans = argotrans
        self.translation_obj = self._prepare_translator(auto_install)

    def _prepare_translator(self, auto_install: bool):
        src_norm = _normalize_lang_code(self.src_lang)
        target_norm = _normalize_lang_code(self.target_lang)

        def _get_installed():
            installed = self.argotrans.get_installed_languages()
            from_l = next((l for l in installed if _normalize_lang_code(getattr(l, 'code', None)) == src_norm), None)
            to_l = next((l for l in installed if _normalize_lang_code(getattr(l, 'code', None)) == target_norm), None)
            if from_l and to_l:
                return from_l.get_translation(to_l)
            return None

        obj = _get_installed()
        if obj:
            return obj

        if not auto_install:
            installed = self.argotrans.get_installed_languages()
            codes = [getattr(l, 'code', None) for l in installed]
            raise RuntimeError(f"Argos: No installed package for {self.src_lang}->{self.target_lang}. Installed: {codes}. Use --auto-install-argos.")

        print(f"Argos: Target pair {self.src_lang}->{self.target_lang} not installed. Searching...")
        self.argopackage.update_package_index()
        available = self.argopackage.get_available_packages()

        candidate = None
        for p in available:
            if (_normalize_lang_code(getattr(p, 'from_code', None)) == src_norm and 
                _normalize_lang_code(getattr(p, 'to_code', None)) == target_norm):
                candidate = p
                break
        
        if not candidate:
            raise RuntimeError(f"Argos: No exact package found in registry for {self.src_lang}->{self.target_lang}.")

        print(f"Argos: Installing {getattr(candidate,'from_code',None)}->{getattr(candidate,'to_code',None)}...")
        candidate.install()

        obj = _get_installed()
        if not obj:
            raise RuntimeError(f"Argos: Installation failed or package not recognized for {self.src_lang}->{self.target_lang}.")
        
        return obj

    def translate(self, text: str) -> str:
        return self.translation_obj.translate(text)

class M2M100Translator(BaseTranslator):
    def __init__(self, src_lang: Optional[str], target_lang: str, model_name: str = "facebook/m2m100_418M", device: str = "cpu", model_cache_dir: Optional[str] = None):
        super().__init__(src_lang, target_lang)
        if not self.src_lang:
            raise RuntimeError("M2M100Translator requires 'src_lang' to be specified.")
            
        from_pretrained_kwargs = {}
        if model_cache_dir:
            hf_dir = os.path.join(model_cache_dir, 'hugging_face')
            os.makedirs(hf_dir, exist_ok=True)
            
            os.environ.setdefault('HF_HOME', hf_dir)
            os.environ.setdefault('TRANSFORMERS_CACHE', hf_dir)
            os.environ.setdefault('TORCH_HOME', hf_dir)
            
            from_pretrained_kwargs['cache_dir'] = hf_dir

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
        except ImportError:
            raise RuntimeError("transformers/torch not installed.")

        # 2. Determine device
        self.device = torch.device(device)
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        
        print(f"M2M100: Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **from_pretrained_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **from_pretrained_kwargs)
        self.model.to(self.device)
        
        self.tokenizer.src_lang = _normalize_lang_code(self.src_lang)
        self.target_lang_id = self.tokenizer.get_lang_id(_normalize_lang_code(self.target_lang))

    def translate(self, text: str) -> str:
        # Move inputs to device
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        # Generate
        generated = self.model.generate(**encoded, forced_bos_token_id=self.target_lang_id)
        # Decode
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def _make_translator(mode, src_lang, target_lang, proxy, auto_install, device, model_cache_dir):
    key = mode.lower()
    if key in ("online", "google"):
        return DeepGoogleTranslator(src_lang, target_lang, proxy), 8
    if key in ("offline", "argos"):
        return ArgoTranslator(src_lang, target_lang, auto_install=auto_install), 4

    models = {
        "m2m100": "facebook/m2m100_418M",
        "m2m100_418m": "facebook/m2m100_418M",
        "m2m100_1.2b": "facebook/m2m100_1.2B",
    }
    if key.startswith("model:"):
        model_name = mode.split(":", 1)[1].strip() or models["m2m100"]
    elif key in models:
        model_name = models[key]
    else:
        raise ValueError(f"Unknown translation mode: {mode}")

    # A single GPU model must only generate for one segment at a time.
    return M2M100Translator(src_lang, target_lang, model_name=model_name,
                            device=device, model_cache_dir=model_cache_dir), 1


def _translation_worker():
    # Keep provider logs (including native code writing to fd 1) off the event pipe.
    with os.fdopen(os.dup(1), "w", encoding="utf-8") as events:
        os.dup2(2, 1)
        lock = threading.Lock()

        def send(*event):
            with lock:
                print(json.dumps([time.monotonic(), *event]), file=events, flush=True)

        try:
            paths, texts, options = json.load(sys.stdin)
            sys.path[:] = paths
            translator, workers = _make_translator(**options)

            def translate(item):
                index, text = item
                try:
                    send("started", index)
                    result = translator.translate(text) if text else ""
                    if text and (not isinstance(result, str) or not result.strip()):
                        raise ValueError("provider returned an empty translation")
                    send("translated", index, result)
                except Exception as exc:
                    send("failed", index, str(exc))

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(translate, enumerate(texts)))
        except Exception as exc:
            send("failed", None, str(exc))


def _collect_translations(events, total, timeout):
    results = [None] * total
    deadlines = {}
    completed = 0

    def timeout_error(index):
        return RuntimeError(f"Translation timed out for segment {index + 1}/{total} after {timeout}s")

    while completed < total:
        first = min(deadlines, key=deadlines.get, default=None)
        remaining = None if first is None else max(0, deadlines[first] - time.monotonic())
        try:
            line = events.get(timeout=remaining)
        except queue.Empty:
            raise timeout_error(first) from None
        if line is None:
            raise RuntimeError("Translation worker exited before returning all segments")

        try:
            timestamp, kind, index, *data = json.loads(line)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid response from translation worker") from exc
        if first is not None and timestamp > deadlines[first]:
            raise timeout_error(first)
        if kind == "failed":
            where = " during setup" if index is None else f" for segment {index + 1}/{total}"
            raise RuntimeError(f"Translation failed{where}: {data[0]}")
        if kind == "started":
            deadlines[index] = timestamp + timeout
        elif kind == "translated":
            del deadlines[index]
            results[index] = data[0]
            completed += 1
            print(f"\rTranslated segment {completed}/{total}", end="", flush=True)
    return results


def translate_segments(
    segments: List[Dict],
    target_lang: str,
    mode: str = 'online',
    proxy: Optional[str] = None,
    timeout: float = 10.0,
    src_lang: Optional[str] = None,
    auto_install: bool = False,
    device: str = 'cpu',
    model_cache_dir: Optional[str] = None
) -> List[Dict]:
    """Translate each segment within its own deadline, preserving order and metadata."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("Translation timeout must be a positive, finite number of seconds")
    if not segments:
        return segments

    texts = [(segment.get("text") or "").strip() for segment in segments]
    options = dict(mode=mode, src_lang=src_lang, target_lang=target_lang, proxy=proxy,
                   auto_install=auto_install, device=device,
                   model_cache_dir=os.fspath(model_cache_dir) if model_cache_dir else None)
    env = os.environ.copy()
    if proxy:
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, http_proxy=proxy, https_proxy=proxy)
    env["PYTHONIOENCODING"] = "utf-8"
    # GoogleTranslator has no request-timeout hook. Keep the provider's HTTP behavior
    # and isolate the batch so a blocked request or native model can be stopped.
    with subprocess.Popen([sys.executable, __file__], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, encoding="utf-8", env=env) as process:
        events = queue.Queue()

        def read_events():
            try:
                for line in process.stdout:
                    events.put(line)
            finally:
                events.put(None)

        reader = threading.Thread(target=read_events, daemon=True)
        reader.start()
        try:
            with process.stdin:
                paths = [path for path in sys.path if isinstance(path, str)]
                json.dump([paths, texts, options], process.stdin)
            results = _collect_translations(events, len(texts), timeout)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        except BrokenPipeError as exc:
            raise RuntimeError("Translation worker exited during setup") from exc
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            reader.join()

    for segment, translated in zip(segments, results):
        segment["translated_text"] = translated
    print()
    return segments


if __name__ == "__main__":
    _translation_worker()
