from typing import List, Dict, Optional, Any
import os
import concurrent.futures
import time
from abc import ABC, abstractmethod
import sys

# --- Proxy Utilities ---

def _use_proxy_env(proxy: Optional[str]):
    """Set HTTP(S)_PROXY env vars, return previous values."""
    prev_http = os.environ.get('HTTP_PROXY')
    prev_https = os.environ.get('HTTPS_PROXY')
    if proxy:
        os.environ['HTTP_PROXY'] = proxy
        os.environ['HTTPS_PROXY'] = proxy
    return prev_http, prev_https


def _restore_proxy_env(prev_http, prev_https):
    if prev_http is None:
        os.environ.pop('HTTP_PROXY', None)
    else:
        os.environ['HTTP_PROXY'] = prev_http
    if prev_https is None:
        os.environ.pop('HTTPS_PROXY', None)
    else:
        os.environ['HTTPS_PROXY'] = prev_https

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
        try:
            return self.translation_obj.translate(text)
        except Exception:
            return str(text or "")

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


# --- Main Logic ---

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
    """
    Translate segments using the specified engine.
    
    Args:
        mode: 'online'/'google', 'offline'/'argos', 'm2m100', or 'model:<repo_id>'.
        device: 'cpu' or 'cuda' (only used for m2m100).
    """
    
    translator: BaseTranslator = None
    mode_raw = mode.lower()
    
    # Mode parsing:
    # 1. Default M2M100 model name
    m2m_model_name = "facebook/m2m100_418M"

    # 2. Check for a 'model:' prefix specifying an explicit model
    if mode_raw.startswith("model:"):
        # Extract the model name (e.g., model:facebook/m2m100_1.2B -> facebook/m2m100_1.2B)
        # Do not lower() the repo id (preserve case)
        parts = mode.split(":", 1)
        if len(parts) > 1 and parts[1].strip():
            m2m_model_name = parts[1].strip()
        # Force the processing mode to 'm2m100'
        mode_raw = "m2m100"

    # 3. If mode is simply 'm2m100', keep the default 418M model

    prev_http, prev_https = _use_proxy_env(proxy)
    
    try:
        # 1. Initialize Translator
        if mode_raw in ['online', 'google']:
            translator = DeepGoogleTranslator(src_lang, target_lang, proxy)
            workers = 8 # IO bound
        elif mode_raw in ['offline', 'argos']:
            translator = ArgoTranslator(src_lang, target_lang, auto_install=auto_install)
            workers = 4 # CPU bound
        elif mode_raw == 'm2m100':
            translator = M2M100Translator(src_lang, target_lang, model_name=m2m_model_name, device=device, model_cache_dir=model_cache_dir)
            # IMPORTANT: Threading with a single GPU model instance causes VRAM contention and errors.
            # We enforce serial execution (workers=1) for M2M100 to ensure safety and stability.
            workers = 1 
        else:
            raise ValueError(f"Unknown translation mode: {mode}")

        total = len(segments)
        
        # 2. Execute Translation
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for i, seg in enumerate(segments):
                text = seg.get('text', '').strip()
                if not text:
                    futures.append(ex.submit(lambda: ""))
                    continue
                
                futures.append(ex.submit(translator.translate, text))

            # 3. Collect Results
            max_msg_len = 0
            for idx, fut in enumerate(futures, start=1):
                try:
                    start = time.time()
                    translated = fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    translated = segments[idx-1].get('text')
                    print(f"Timeout segment {idx}/{total} (>{timeout}s)")
                except Exception as e:
                    translated = segments[idx-1].get('text')
                    print(f"Error segment {idx}/{total}: {e}")

                if translated is None:
                    translated = ''
                else:
                    translated = str(translated)

                segments[idx-1]['translated_text'] = translated
                
                # Update progress on the same line without emitting a newline.
                try:
                    msg = f"Translated segment {idx}/{total}"
                    # track max length so we can clear remnants of previous longer messages
                    if len(msg) > max_msg_len:
                        max_msg_len = len(msg)
                    # write carriage return + message + padding to clear previous content
                    sys.stdout.write('\r' + msg + ' ' * (max_msg_len - len(msg)))
                    sys.stdout.flush()
                except Exception:
                    # best-effort fallback: try print with carriage return, then as last resort print without newline
                    try:
                        print(f"Translated segment {idx}/{total}", end='\r', flush=True)
                    except Exception:
                        print(f"Translated segment {idx}/{total}", end='')

    finally:
        _restore_proxy_env(prev_http, prev_https)

    try:
        print()
    except Exception:
        pass

    return segments
