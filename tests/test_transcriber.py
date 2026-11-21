import importlib
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch


class TestTranscriber(unittest.TestCase):
    def setUp(self):
        self.raw = [{"start": 0, "end": 1, "text": "bonjour"}]
        self.aligned = [dict(self.raw[0], words=[{"word": "bonjour"}])]
        self.asr = Mock()
        self.asr.transcribe.return_value = {"segments": self.raw, "language": "fr"}
        self.whisper = Mock()
        self.whisper.load_model.return_value = self.asr
        self.whisperx = Mock()
        self.whisperx.load_model.return_value = self.asr
        self.whisperx.load_align_model.return_value = (object(), {})
        self.whisperx.align.return_value = {"segments": self.aligned}
        self.cuda = Mock()
        self.cuda.is_available.return_value = False
        modules = {"torch": SimpleNamespace(cuda=self.cuda),
                   "whisper": self.whisper, "whisperx": self.whisperx}
        backends = patch.dict(sys.modules, modules)
        backends.start()
        self.addCleanup(backends.stop)
        self.module = importlib.import_module("video_translator.transcriber")

    def test_module_can_be_imported_without_torch(self):
        script = "import sys\nsys.modules['torch'] = None\nimport video_translator.transcriber"
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_automatic_cpu_uses_supported_compute_type(self):
        segments, device = self.module.transcribe("audio.wav", mode="whisperx")
        self.assertEqual((segments, device), (self.aligned, "cpu"))
        self.assertEqual(self.whisperx.load_model.call_args.kwargs["compute_type"], "int8")

    def test_asr_uses_prefetched_cache(self):
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ):
            self.module.transcribe("audio.wav", mode="whisperx", model_cache_dir=cache)
        options = self.whisperx.load_model.call_args.kwargs
        self.assertEqual(options.get("download_root"), os.path.join(cache, "hugging_face"))

    def test_indexed_cuda_passes_separate_index_and_preserves_alignment_device(self):
        segments = self.module.Transcriber(device="cuda:2").transcribe("audio.wav")
        self.assertEqual(segments, self.aligned)
        args, options = self.whisperx.load_model.call_args
        self.assertEqual(args[:2], ("small", "cuda"))
        self.assertEqual(options["device_index"], 2)
        self.assertEqual(options["compute_type"], "float16")
        self.assertEqual(self.whisperx.load_align_model.call_args.kwargs["device"], "cuda:2")
        self.assertEqual(self.whisperx.align.call_args.args[4], "cuda:2")

    def test_explicit_compute_type_overrides_default(self):
        self.module.transcribe("audio.wav", device="cpu", compute_type="float32", mode="whisperx")
        self.assertEqual(self.whisperx.load_model.call_args.kwargs["compute_type"], "float32")

    def test_language_survives_all_backends_and_alignment_failure(self):
        self.assertTrue(hasattr(self.module, "transcribe_with_language"))
        for mode in ("whisper", "mixed", "whisperx"):
            with self.subTest(mode=mode):
                expected = self.raw if mode == "whisper" else self.aligned
                self.assertEqual(
                    self.module.transcribe_with_language("audio.wav", device="cpu", mode=mode),
                    (expected, "cpu", "fr"),
                )
        self.whisperx.align.side_effect = RuntimeError("no alignment")
        self.assertEqual(
            self.module.transcribe_with_language("audio.wav", device="cpu"),
            (self.raw, "cpu", "fr"),
        )

    def test_owned_models_collected_before_next_stage_and_cuda_cache_release(self):
        class Model:
            def __init__(self):
                self.cycle = self

            def transcribe(self, *args, **kwargs):
                return {"segments": [], "language": "fr"}

        for mode in ("mixed", "whisperx", "whisper"):
            with self.subTest(mode=mode):
                references = []
                live_at_alignment = []
                live_at_cache_release = []

                def load_model(*args, **kwargs):
                    model = Model()
                    references.append(weakref.ref(model))
                    return model

                def load_alignment(**kwargs):
                    live_at_alignment.append(references[0]() is not None)
                    return load_model(), {}

                def empty_cache():
                    live_at_cache_release.append(any(ref() is not None for ref in references))

                self.whisper.load_model.side_effect = load_model
                self.whisperx.load_model.side_effect = load_model
                self.whisperx.load_align_model.side_effect = load_alignment
                # A Mock would retain the alignment model in its call history.
                self.whisperx.align = lambda *args, **kwargs: {"segments": []}
                self.cuda.empty_cache.side_effect = empty_cache
                self.module.transcribe("audio.wav", mode=mode, device="cuda:1")
                self.assertEqual(live_at_alignment, [] if mode == "whisper" else [False])
                self.assertEqual(live_at_cache_release, [False] if mode == "whisper" else [False, False])

    def test_external_model_can_be_reused(self):
        for _ in range(2):
            segments, device = self.module.transcribe(
                "audio.wav", mode="whisper", device="cuda", external_model=self.asr,
            )
            self.assertEqual((segments, device), (self.raw, "cuda"))
        self.assertEqual(self.asr.transcribe.call_count, 2)
        self.whisper.load_model.assert_not_called()
        self.cuda.empty_cache.assert_not_called()
