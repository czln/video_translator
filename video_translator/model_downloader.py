import os
import json
import time
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from typing import Iterable, List, Dict, Any, Optional

# NOTE: Do not import `huggingface_hub` at module scope,
# otherwise it may initialize before we set environment variables and
# the HF mirror/endpoint settings may not take effect.

MANIFEST_FILENAME = "model_manifest.json"

DEFAULT_MANIFEST = {
    "last_updated": None,
    # Default manifest mappings
    "whisper_map": {
        "tiny": "Systran/faster-whisper-tiny",
        "base": "Systran/faster-whisper-base",
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large-v2": "Systran/faster-whisper-large-v2",
        # large-v3 will be discovered via update, or handled by fallback logic
    },
    "alignment_repos": [
        "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    ],
    "translation_repos": [
        "facebook/m2m100_418M",
        "facebook/m2m100_1.2B"
    ]
}

class ManifestManager:
    """Manage loading, saving, and network updates for the model manifest file."""
    def __init__(self, root_dir: str):
        self.filepath = os.path.join(root_dir, MANIFEST_FILENAME)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Simple validation: reset to default if the format is too old
                    if "whisper_map" not in data: 
                        return DEFAULT_MANIFEST.copy()
                    return data
            except Exception as e:
                print(f"Warning: Failed to load manifest {self.filepath}: {e}")
        return DEFAULT_MANIFEST.copy()

    def save(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            print(f"Manifest saved to: {self.filepath}")
        except Exception as e:
            print(f"Error saving manifest: {e}")

    def update_from_network(self, proxy: str = None, mirror: str = None):
        """Connect to the Hugging Face API to discover new models and update the manifest."""
        print("Connecting to Hugging Face API to update model list...")
        
        # 1. Set environment variables (must be done before importing HfApi)
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
        if mirror:
            mirror = mirror.rstrip('/')
            os.environ["HF_ENDPOINT"] = mirror
            print(f"Using Mirror: {mirror}")

        try:
            # 2. Delay imports until after environment is configured
            from huggingface_hub import HfApi
            api = HfApi()

            # Update Faster-Whisper mappings
            print("Searching for Systran/faster-whisper models...")
            models = api.list_models(author="Systran", search="faster-whisper", limit=50)
            
            new_map = {}
            for m in models:
                # modelId example: "Systran/faster-whisper-large-v3"
                if "faster-whisper-" in m.modelId:
                    key = m.modelId.split("faster-whisper-")[-1]
                    new_map[key] = m.modelId
            
            if new_map:
                current_map = self.data.get("whisper_map", {})
                current_map.update(new_map)
                self.data["whisper_map"] = current_map
                print(f"Updated Whisper map: {list(new_map.keys())}")

            # Update M2M100 model list
            print("Searching for facebook/m2m100 models...")
            m2m_models = api.list_models(author="facebook", search="m2m100", limit=20)
            new_m2m = []
            for m in m2m_models:
                if "m2m100" in m.modelId and "ggml" not in m.modelId.lower():
                    new_m2m.append(m.modelId)
            
            if new_m2m:
                self.data["translation_repos"] = sorted(list(set(new_m2m)))
                print(f"Updated Translation repos: {len(new_m2m)} found.")

            self.data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save()
            print("Update successful!")

        except ImportError:
            print("Error: `huggingface_hub` not installed.")
        except Exception as e:
            print(f"Error updating from network: {e}")

    # --- Properties added previously when an error occurred ---
    # These should remain indented inside the ManifestManager class
    
    @property
    def whisper_map(self) -> Dict[str, str]:
        return self.data.get("whisper_map", DEFAULT_MANIFEST["whisper_map"])

    @property
    def whisper_sizes(self) -> List[str]:
        keys = list(self.whisper_map.keys())
        # Custom priority ordering for whisper sizes
        priority = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
        res = [k for k in priority if k in keys]
        res += [k for k in keys if k not in priority]
        return res

    @property
    def translation_repos(self) -> List[str]:
        return self.data.get("translation_repos", DEFAULT_MANIFEST["translation_repos"])

    @property
    def alignment_repos(self) -> List[str]:
        return self.data.get("alignment_repos", DEFAULT_MANIFEST["alignment_repos"])


class ModelDownloader:
    def __init__(self, proxy: str = None, hf_mirror: str = None):
        self.proxy = proxy
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

        # 设置环境变量
        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
        
        if hf_mirror:
            hf_mirror = hf_mirror.rstrip('/')
            os.environ["HF_ENDPOINT"] = hf_mirror
            print(f"Hugging Face mirror set as: {hf_mirror}")

    def download_all_models(
        self, 
        save_root: str = "./model-cache", 
        model_size: str = "small",
        include_openai_whisper: bool = True,
        include_faster_whisper: bool = True,
        include_alignment: bool = True,
        include_m2m100: bool = True,
        specific_translation_repo: Optional[str] = None 
    ) -> List[str]:
        
        results = []
        os.makedirs(save_root, exist_ok=True)
        
        # Load the manifest
        manifest = ManifestManager(save_root)
        print(f"Using manifest from: {manifest.filepath}")

        whisper_dir = os.path.join(save_root, 'whisper')
        hf_dir = os.path.join(save_root, 'hugging_face')
        os.makedirs(whisper_dir, exist_ok=True)
        os.makedirs(hf_dir, exist_ok=True)

        # A. OpenAI Whisper
        if include_openai_whisper:
            try:
                import whisper
                print(f"Downloading OpenAI Whisper ({model_size})...")
                model = whisper.load_model(model_size, download_root=whisper_dir)
                del model
                results.append(f"OpenAI Whisper: {model_size}")
            except Exception as e:
                print(f"Failed to download OpenAI Whisper: {e}")

        # Sections B/C/D use a shared helper function
        
        def _do_download(repo_id):
            print(f"Downloading HF Repo: {repo_id}")
            try:
                # Delay import so HF_ENDPOINT is applied before using the HF API
                from huggingface_hub import snapshot_download
                path = snapshot_download(
                    repo_id=repo_id, 
                    cache_dir=hf_dir, 
                    repo_type='model', 
                    proxies=self.proxies,
                    resume_download=True
                )
                results.append(path)
            except Exception as ex:
                print(f"Failed to download {repo_id}: {ex}")
                results.append(None)

        # B. Faster-Whisper
        if include_faster_whisper:
            fw_repo = manifest.whisper_map.get(model_size)
            if not fw_repo:
                fw_repo = f"Systran/faster-whisper-{model_size}"
                print(f"Warning: {model_size} not in manifest. Trying: {fw_repo}")
            _do_download(fw_repo)

        # C. Alignment
        if include_alignment:
            for repo in manifest.alignment_repos:
                _do_download(repo)

        # D. Translation
        if include_m2m100:
            repos_to_dl = [specific_translation_repo] if specific_translation_repo else manifest.translation_repos[:1]
            for repo in repos_to_dl:
                _do_download(repo)

        return results

    # 通用下载方法 (保留，但也修复延迟引用)
    def download(self, source, save_dir="./models", filename=None, repo_type="model"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if source.startswith("http://") or source.startswith("https://"):
            return self._download_from_url(source, save_dir, filename)
        else:
            return self._download_from_hf(source, filename, save_dir, repo_type)

    def _download_from_url(self, url, save_dir, filename=None):
        if not filename:
            filename = os.path.basename(urlparse(url).path)
        file_path = os.path.join(save_dir, filename)
        first_byte = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        header = {"Range": f"bytes={first_byte}-"}
        try:
            print(f"Downloading URL: {url}")
            response = requests.get(url, headers=header, stream=True, proxies=self.proxies, timeout=60)
            if response.status_code == 416: 
                return file_path
            total_size = int(response.headers.get('content-length', 0)) + first_byte
            mode = 'ab' if first_byte > 0 else 'wb'
            with open(file_path, mode) as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, initial=first_byte) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return file_path
        except Exception as e:
            print(f"URL Download failed: {e}")
            return None

    def _download_from_hf(self, repo_id, filename, save_dir, repo_type):
        endpoint = os.getenv('HF_ENDPOINT', 'original web site')
        print(f"Downloading from Hugging Face ({endpoint}): {repo_id}...")
        
        # 延迟引用
        from huggingface_hub import hf_hub_download, snapshot_download
        
        try:
            if filename:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    local_dir=save_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    proxies=self.proxies
                )
                return path
            else:
                path = snapshot_download(
                    repo_id=repo_id, 
                    revision="main", 
                    cache_dir=save_dir, 
                    repo_type=repo_type, 
                    proxies=self.proxies
                )
                return path
        except Exception as e:
            print(f"HF Download failed: {e}")
            return None
