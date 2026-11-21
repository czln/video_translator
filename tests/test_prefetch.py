import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from video_translator.model_downloader import ManifestManager, ModelDownloader


class PrefetchTests(unittest.TestCase):
    def test_help_without_download_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "video_translator.prefetch", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-cache-dir", result.stdout)

    def test_prefetch_and_update_commands_exit_nonzero_on_failure(self):
        with TemporaryDirectory() as root:
            for args, answers in [(["--update"], ""), ([], "\n\nn\nn\nn\n")]:
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, "-S", "-m", "video_translator.prefetch",
                         "--model-cache-dir", root, *args],
                        input=answers,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("Done.", result.stdout)
                    self.assertTrue(result.stderr.strip())

    def test_explicit_proxy_overrides_existing_lowercase_proxy(self):
        from urllib.request import getproxies_environment

        api = SimpleNamespace(list_models=lambda **kwargs: [])
        hub = SimpleNamespace(HfApi=lambda: api)
        for command in ("prefetch", "update"):
            with self.subTest(command=command), TemporaryDirectory() as root, patch.dict(
                os.environ, {"http_proxy": "old", "https_proxy": "old"}
            ), patch.dict(sys.modules, {"huggingface_hub": hub}):
                if command == "prefetch":
                    ModelDownloader(proxy="http://selected")
                else:
                    ManifestManager(root).update_from_network(proxy="http://selected")
                self.assertEqual(getproxies_environment()["https"], "http://selected")

    def test_all_selected_failures_are_reported(self):
        modules = {
            "whisper": SimpleNamespace(load_model=Mock(side_effect=OSError("Whisper failure"))),
            "whisperx": SimpleNamespace(load_align_model=Mock(side_effect=OSError("align failure"))),
            "huggingface_hub": SimpleNamespace(snapshot_download=Mock(side_effect=OSError("Hub failure"))),
        }
        with TemporaryDirectory() as root, patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError) as failure:
                ModelDownloader().download_all_models(root)
            for message in ["Whisper failure", "align failure", "Systran/faster-whisper-small",
                            "facebook/m2m100_418M", "Hub failure"]:
                self.assertIn(message, str(failure.exception))

    def test_default_alignment_uses_runtime_loader_and_snapshots_share_cache(self):
        align = Mock()
        downloads = []

        def snapshot_download(repo_id, cache_dir, repo_type):
            downloads.append((repo_id, cache_dir))
            return os.path.join(cache_dir, repo_id)

        modules = {
            "whisperx": SimpleNamespace(load_align_model=align),
            "huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download),
        }
        with TemporaryDirectory() as root, patch.dict(sys.modules, modules):
            cache = os.path.join(root, "hugging_face")
            ModelDownloader().download_all_models(root, include_openai_whisper=False)
            align.assert_called_once_with(language_code="en", device="cpu", model_dir=cache)
            self.assertEqual(downloads, [
                ("Systran/faster-whisper-small", cache), ("facebook/m2m100_418M", cache),
            ])

    def test_unlisted_whisper_alias_uses_runtime_resolution(self):
        download = Mock(return_value="cached")
        modules = {"faster_whisper.utils": SimpleNamespace(download_model=download)}
        with TemporaryDirectory() as root, patch.dict(sys.modules, modules):
            results = ModelDownloader().download_all_models(
                root, "large", include_openai_whisper=False, include_alignment=False, include_m2m100=False
            )
            download.assert_called_once_with("large", cache_dir=os.path.join(root, "hugging_face"))
            self.assertEqual(results, ["cached"])

    def test_existing_and_refreshed_manifests_keep_runtime_translation_default(self):
        small, large = "facebook/m2m100_418M", "facebook/m2m100_1.2B"
        api = Mock()
        api.list_models.side_effect = [[], [SimpleNamespace(id=r, modelId=r) for r in (large, small)]]
        hub = SimpleNamespace(HfApi=lambda: api, snapshot_download=Mock(return_value="cached"))
        with TemporaryDirectory() as root, patch.dict(sys.modules, {"huggingface_hub": hub}):
            path = Path(root, "model_manifest.json")
            path.write_text(json.dumps({"whisper_map": {}, "translation_repos": [large, small]}))
            for refreshed in [False, True]:
                with self.subTest(refreshed=refreshed):
                    manifest = ManifestManager(root)
                    if refreshed:
                        manifest.update_from_network()
                        manifest = ManifestManager(root)
                    self.assertEqual(manifest.translation_repos[0], small)
                    ModelDownloader().download_all_models(
                        root, include_openai_whisper=False, include_faster_whisper=False, include_alignment=False,
                    )
                    self.assertEqual(hub.snapshot_download.call_args.kwargs["repo_id"], small)

    def test_update_propagates_manifest_write_failure(self):
        hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(list_models=lambda **kwargs: []))
        with TemporaryDirectory() as root, patch.dict(sys.modules, {"huggingface_hub": hub}):
            manifest = ManifestManager(root)
            manifest.save()
            path = Path(manifest.filepath)
            previous = path.read_bytes()

            def write_failure(data, stream, **kwargs):
                stream.write("{")
                raise OSError("Disk full")

            with patch.object(json, "dump", side_effect=write_failure), self.assertRaises(OSError):
                manifest.update_from_network()
            self.assertEqual(path.read_bytes(), previous)


class URLDownloadTests(unittest.TestCase):
    def download(self, root, status, headers, body):
        response = MagicMock(status_code=status, headers=headers)
        response.__enter__.return_value = response
        response.iter_content.return_value = [body]
        if status >= 400:
            response.raise_for_status.side_effect = OSError(f"HTTP {status}")
        modules = {
            "requests": SimpleNamespace(get=Mock(return_value=response)),
            "tqdm": SimpleNamespace(tqdm=MagicMock()),
        }
        with patch.dict(sys.modules, modules):
            return ModelDownloader().download("https://example.test/model", root)

    def test_ignored_range_replaces_existing_file(self):
        with TemporaryDirectory() as root:
            path = Path(root, "model")
            path.write_bytes(b"fi")
            self.assertEqual(self.download(root, 200, {}, b"file"), str(path))
            self.assertEqual(path.read_bytes(), b"file")

    def test_http_errors_and_mismatched_ranges_preserve_existing_file(self):
        with TemporaryDirectory() as root:
            path = Path(root, "model")
            path.write_bytes(b"fi")
            for status, headers in [(404, {}), (206, {"Content-Range": "bytes 1-3/4"}),
                                    (416, {"Content-Range": "bytes */1"})]:
                with self.subTest(status=status):
                    self.assertIsNone(self.download(root, status, headers, b"bad"))
                    self.assertEqual(path.read_bytes(), b"fi")
            self.assertEqual(self.download(root, 206, {"Content-Range": "bytes 2-3/4"}, b"le"), str(path))
            self.assertEqual(path.read_bytes(), b"file")
            self.assertEqual(self.download(root, 416, {"Content-Range": "bytes */4"}, b""), str(path))
