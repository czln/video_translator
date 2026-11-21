import json
import os
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

# NOTE: Do not import `huggingface_hub` at module scope,
# otherwise it may initialize before we set environment variables and
# the HF mirror/endpoint settings may not take effect.

MANIFEST_FILENAME = "model_manifest.json"
DEFAULT_TRANSLATION_REPO = "facebook/m2m100_418M"

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
    # Additional snapshots beyond the default English model selected by WhisperX.
    "alignment_repos": [],
    "translation_repos": [
        DEFAULT_TRANSLATION_REPO,
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
                        return deepcopy(DEFAULT_MANIFEST)
                    return data
            except Exception as e:
                print(f"Warning: Failed to load manifest {self.filepath}: {e}")
        return deepcopy(DEFAULT_MANIFEST)

    def save(self):
        with TemporaryDirectory(dir=os.path.dirname(self.filepath) or '.') as scratch:
            temporary = os.path.join(scratch, MANIFEST_FILENAME)
            with open(temporary, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            os.replace(temporary, self.filepath)
        print(f"Manifest saved to: {self.filepath}")

    def update_from_network(self, proxy: str = None, mirror: str = None):
        """Connect to the Hugging Face API to discover new models and update the manifest."""
        print("Connecting to Hugging Face API to update model list...")
        
        # 1. Set environment variables (must be done before importing HfApi)
        if proxy:
            os.environ.update(http_proxy=proxy, https_proxy=proxy, HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
        if mirror:
            mirror = mirror.rstrip('/')
            os.environ["HF_ENDPOINT"] = mirror
            print(f"Using Mirror: {mirror}")

        from huggingface_hub import HfApi
        api = HfApi()

        def model_ids(author, search, limit):
            for model in api.list_models(author=author, search=search, limit=limit):
                yield getattr(model, "id", None) or model.modelId

        print("Searching for Systran/faster-whisper models...")
        prefix = "Systran/faster-whisper-"
        for repo in model_ids("Systran", "faster-whisper", 50):
            if repo.startswith(prefix):
                self.data["whisper_map"][repo[len(prefix):]] = repo

        print("Searching for facebook/m2m100 models...")
        discovered = {
            repo for repo in model_ids("facebook", "m2m100", 20)
            if "m2m100" in repo and "ggml" not in repo.lower()
        }
        if discovered:
            self.data["translation_repos"] = [DEFAULT_TRANSLATION_REPO] + sorted(discovered - {DEFAULT_TRANSLATION_REPO})

        self.data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        print("Update successful!")

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
        repos = self.data.get("translation_repos", DEFAULT_MANIFEST["translation_repos"])
        # Older generated manifests sorted 1.2B before the runtime default.
        return sorted(repos, key=lambda repo: repo != DEFAULT_TRANSLATION_REPO)

    @property
    def alignment_repos(self) -> List[str]:
        return self.data.get("alignment_repos", DEFAULT_MANIFEST["alignment_repos"])


class ModelDownloader:
    def __init__(self, proxy: str = None, hf_mirror: str = None):
        self.proxy = proxy
        self.proxies = {"http": proxy, "https": proxy} if proxy else None

        # 设置环境变量
        if proxy:
            os.environ.update(http_proxy=proxy, https_proxy=proxy, HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
        
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

        failures = []

        @contextmanager
        def attempt(label):
            print(f"Downloading {label}...")
            try:
                yield
            except Exception as exc:
                failures.append(f"{label}: {exc}")

        def download_snapshot(repo_id):
            with attempt(repo_id):
                from huggingface_hub import snapshot_download
                results.append(snapshot_download(repo_id=repo_id, cache_dir=hf_dir, repo_type="model"))

        if include_openai_whisper:
            label = f"OpenAI Whisper: {model_size}"
            with attempt(label):
                import whisper
                whisper.load_model(model_size, device="cpu", download_root=whisper_dir)
                results.append(label)

        if include_faster_whisper:
            repo_id = manifest.whisper_map.get(model_size)
            if repo_id:
                download_snapshot(repo_id)
            else:
                # Faster Whisper resolves aliases such as "large" using its installed model map.
                with attempt(f"Faster Whisper: {model_size}"):
                    from faster_whisper.utils import download_model
                    results.append(download_model(model_size, cache_dir=hf_dir))

        if include_alignment:
            label = "WhisperX English alignment"
            with attempt(label):
                import whisperx
                whisperx.load_align_model(language_code="en", device="cpu", model_dir=hf_dir)
                results.append(label)
            for repo in manifest.alignment_repos:
                download_snapshot(repo)

        if include_m2m100:
            download_snapshot(specific_translation_repo or DEFAULT_TRANSLATION_REPO)

        if failures:
            raise RuntimeError("Prefetch failed:\n" + "\n".join(failures))
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
        import requests
        from tqdm import tqdm

        if not filename:
            filename = os.path.basename(urlparse(url).path)
        file_path = os.path.join(save_dir, filename)
        first_byte = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        headers = {"Accept-Encoding": "identity"}
        if first_byte:
            headers["Range"] = f"bytes={first_byte}-"
        try:
            print(f"Downloading URL: {url}")
            with requests.get(url, headers=headers, stream=True, proxies=self.proxies, timeout=60) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status_code == 416 and os.path.isfile(file_path) and content_range == f"bytes */{first_byte}":
                    return file_path
                response.raise_for_status()

                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match:
                        raise ValueError(f"Invalid Content-Range: {content_range!r}")
                    start, end, total_size = map(int, match.groups())
                    if start != first_byte or end < start or end != total_size - 1:
                        raise ValueError(f"Unexpected Content-Range: {content_range!r}")
                elif response.status_code == 200:
                    # The server may ignore Range and return the entire file.
                    first_byte = 0
                    length = response.headers.get("Content-Length")
                    total_size = int(length) if length and not response.headers.get("Content-Encoding") else None
                else:
                    raise ValueError(f"Unexpected HTTP status {response.status_code}")

                with open(file_path, "ab" if first_byte else "wb") as f:
                    with tqdm(total=total_size, unit="B", unit_scale=True, initial=first_byte) as progress:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                            progress.update(len(chunk))
                    if total_size is not None and f.tell() != total_size:
                        raise OSError(f"Incomplete download: expected {total_size} bytes, got {f.tell()}")
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
                    local_dir=save_dir
                )
                return path
            else:
                path = snapshot_download(
                    repo_id=repo_id, 
                    revision="main", 
                    cache_dir=save_dir, 
                    repo_type=repo_type
                )
                return path
        except Exception as e:
            print(f"HF Download failed: {e}")
            return None
