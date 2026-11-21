from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from video_translator import cli
from video_translator.srt import _format_timestamp


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / "meeting.mp4"
        self.source.write_bytes(b"first")
        self.cache = self.root / "cache"
        self.stack.enter_context(redirect_stdout(StringIO()))
        self.stack.enter_context(redirect_stderr(StringIO()))
        self.extract = self.stack.enter_context(patch.object(cli, "extract_audio", side_effect=self.extract_sample))
        self.recognize = self.stack.enter_context(patch.object(cli, "transcribe_with_language",
                                                        side_effect=self.recognize_sample))

    @staticmethod
    def extract_sample(source, wav):
        Path(wav).write_bytes(Path(source).read_bytes())

    @staticmethod
    def recognize_sample(wav, **options):
        return [{"text": Path(wav).read_text(), "start": 0, "end": 1}], "cpu", "en"

    def invoke(self, *args, source=None):
        argv = ["video-translator", str(source or self.source), "--device", "cpu", *map(str, args)]
        with patch.object(sys, "argv", argv):
            cli.main()

    def test_default_extracts_audio_and_removes_temp_file(self):
        self.invoke()
        self.extract.assert_called_once()
        self.assertFalse(Path(self.extract.call_args.args[1]).exists())
        self.assertIn("first", self.source.with_suffix(".srt").read_text())

    def test_force_reextracts_even_when_source_metadata_is_unchanged(self):
        self.invoke("--cache-dir", self.cache)
        self.invoke("--cache-dir", self.cache, "--force")
        self.assertEqual(self.extract.call_count, 2)

    def test_audio_is_reused_across_models_but_not_after_replacing_source(self):
        self.invoke("--cache-dir", self.cache)
        self.invoke("--cache-dir", self.cache, "--model", "tiny")
        self.assertEqual(self.extract.call_count, 1)
        self.source.write_bytes(b"replaced video")
        self.invoke("--cache-dir", self.cache)
        self.assertEqual(self.extract.call_count, 2)
        self.assertIn("replaced video", self.source.with_suffix(".srt").read_text())

    def test_shared_cache_distinguishes_same_name_and_mtime(self):
        second = self.root / "other" / self.source.name
        second.parent.mkdir()
        second.write_bytes(b"other")
        stat = self.source.stat()
        os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.invoke("--cache-dir", self.cache)
        self.invoke("--cache-dir", self.cache, source=second)
        self.assertIn("other", second.with_suffix(".srt").read_text())
        self.assertEqual(self.extract.call_count, 2)

    def test_partial_extraction_is_never_reused(self):
        def interrupt(source, wav):
            Path(wav).write_bytes(b"incomplete")
            raise RuntimeError("interrupted")
        self.extract.side_effect = interrupt
        with self.assertRaises(RuntimeError):
            self.invoke("--cache-dir", self.cache)
        self.extract.side_effect = self.extract_sample
        self.invoke("--cache-dir", self.cache)
        self.assertIn("first", self.source.with_suffix(".srt").read_text())

    def test_rejects_output_that_aliases_source_before_extraction(self):
        symlink = self.root / "link.srt"
        symlink.symlink_to(self.source)
        hardlink = self.root / "hardlink.srt"
        os.link(self.source, hardlink)
        for output in (self.source, symlink, hardlink):
            with self.subTest(output=output), self.assertRaises(SystemExit):
                self.invoke("-o", output)
            self.assertEqual(self.source.read_bytes(), b"first")
        self.extract.assert_not_called()

    def test_online_translator_keeps_provider_auto_detection(self):
        self.recognize.side_effect = None
        self.recognize.return_value = ([{"text": "中文", "start": 0, "end": 1}], "cpu", "zh")
        with patch.object(cli, "translate_segments", return_value=[]) as translate:
            self.invoke("--lang", "fr")
        self.assertIsNone(translate.call_args.kwargs["src_lang"])

    def test_cached_detected_language_reaches_offline_translation(self):
        self.invoke()
        with patch.object(cli, "translate_segments", return_value=[]) as translate:
            self.invoke("--lang", "fr", "--translator-mode", "offline")
        self.recognize.assert_called_once()
        self.assertEqual(translate.call_args.kwargs["src_lang"], "en")

    def test_failed_translation_retries_from_cached_transcription(self):
        with patch.object(cli, "translate_segments", side_effect=RuntimeError("offline")):
            with self.assertRaises(SystemExit):
                self.invoke("--lang", "fr")
        self.assertFalse(self.source.with_suffix(".fr.srt").exists())
        with patch.object(cli, "translate_segments", return_value=[]) as translate:
            self.invoke("--lang", "fr")
            self.invoke("--lang", "fr")
        self.assertEqual(translate.call_count, 1)
        self.recognize.assert_called_once()

    def test_new_transcript_invalidates_old_translation_after_forced_failure(self):
        with patch.object(cli, "translate_segments", return_value=[]):
            self.invoke("--lang", "fr")
        self.recognize.side_effect = None
        self.recognize.return_value = ([{"text": "corrected", "start": 0, "end": 1}], "cpu", "en")
        with patch.object(cli, "translate_segments", side_effect=RuntimeError("offline")):
            with self.assertRaises(SystemExit):
                self.invoke("--force", "--lang", "fr")
        with patch.object(cli, "translate_segments", return_value=[]) as translate:
            self.invoke("--lang", "fr")
        translate.assert_called_once()
        self.assertEqual(translate.call_args.args[0][0]["text"], "corrected")

    def test_legacy_fallback_translation_is_rebuilt(self):
        legacy = self.root / "meeting.mp4.translated.online.fr.small.mixed.json"
        legacy.write_text(json.dumps({"source_mtime": self.source.stat().st_mtime, "target_lang": "fr",
                                      "segments": [{"text": "first", "translated_text": "first"}]}))
        with patch.object(cli, "translate_segments", return_value=[]) as translate:
            self.invoke("--lang", "fr")
        translate.assert_called_once()

    def test_download_only_does_not_require_video_and_honors_model(self):
        argv = ["video-translator", "--download-models", "--model-cache-dir", str(self.cache), "--model", "tiny"]
        with patch.object(sys, "argv", argv), patch.object(cli.ModelDownloader, "download_all_models",
                                                         autospec=True, return_value=[]) as download:
            cli.main()
        self.assertEqual(download.call_args.kwargs, {"save_root": str(self.cache), "model_size": "tiny"})

    def test_millisecond_precision_and_carry(self):
        self.assertEqual(_format_timestamp(1.001), "00:00:01,001")
        self.assertEqual(_format_timestamp(59.9996), "00:01:00,000")
