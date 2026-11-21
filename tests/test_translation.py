import os
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from video_translator.translator import translate_segments


class TranslationTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.modules = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(sys, "path", [str(self.modules), *sys.path]))
        self.write_module("deep_translator.py", """
            import os
            import time

            class GoogleTranslator:
                def __init__(self, source, target):
                    self.source = source
                    self.target = target

                def translate(self, text):
                    if text == 'fail':
                        raise OSError('provider unavailable')
                    if text == 'hang':
                        with open(os.environ['TRANSLATION_TEST_PID'], 'w') as pid_file:
                            pid_file.write(str(os.getpid()))
                        time.sleep(3)
                    if text == 'late':
                        time.sleep(0.36)
                    if text == 'empty':
                        return None
                    if text == 'slow':
                        time.sleep(0.18)
                    if text == 'proxy':
                        from urllib.request import getproxies_environment
                        return getproxies_environment()['https']
                    return f'{self.source}/{self.target}/{text}'
        """)

    def write_module(self, name, code):
        path = self.modules / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(code), encoding="utf-8")

    def test_online_success_keeps_order_and_uses_proxy(self):
        segments = [{"text": " slow "}, {"text": "  "}, {"text": "proxy"}]
        before = os.environ.get("HTTPS_PROXY")
        with patch.dict(os.environ, https_proxy="http://old.invalid:3128"):
            result = translate_segments(segments, "fr", proxy="http://example.invalid:3128")
            self.assertEqual(os.environ["https_proxy"], "http://old.invalid:3128")
        self.assertIs(result, segments)
        self.assertEqual([s["translated_text"] for s in result],
                         ["auto/fr/slow", "", "http://example.invalid:3128"])
        self.assertEqual(os.environ.get("HTTPS_PROXY"), before)

    def test_provider_failure_aborts_without_partial_results(self):
        segments = [{"text": "first"}, {"text": "fail"}]
        with self.assertRaisesRegex(RuntimeError, "segment 2/2.*provider unavailable"):
            translate_segments(segments, "fr")
        self.assertTrue(all("translated_text" not in s for s in segments))

    def test_missing_translation_for_nonempty_text_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "segment 1/1"):
            translate_segments([{"text": "empty"}], "fr")

    def test_hung_provider_exits_promptly(self):
        pid_file = self.modules / "worker.pid"
        started = time.monotonic()
        with patch.dict(os.environ, TRANSLATION_TEST_PID=str(pid_file)):
            with self.assertRaisesRegex(RuntimeError, "segment 1/1.*0.1"):
                translate_segments([{"text": "hang"}], "fr", timeout=0.1)
        self.assertLess(time.monotonic() - started, 2)
        if os.name == "posix":
            pid = int(pid_file.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_overlapping_segments_keep_their_original_deadlines(self):
        segments = [{"text": "slow"}, {"text": "late"}]
        with self.assertRaisesRegex(RuntimeError, "segment 2/2.*0.25"):
            translate_segments(segments, "fr", timeout=0.25)

    def test_queue_wait_does_not_consume_a_segments_deadline(self):
        segments = [{"text": "slow"} for _ in range(16)]
        translate_segments(segments, "fr", timeout=0.30)
        self.assertEqual([s["translated_text"] for s in segments], ["auto/fr/slow"] * 16)

    def test_argos_translation_failure_reaches_caller(self):
        self.write_module("argostranslate/__init__.py", "")
        self.write_module("argostranslate/package.py", "")
        self.write_module("argostranslate/translate.py", """
            class Language:
                def __init__(self, code):
                    self.code = code

                def get_translation(self, target):
                    return self

                def translate(self, text):
                    raise ValueError('Argos inference failed')

            def get_installed_languages():
                return [Language('en'), Language('fr')]
        """)
        with self.assertRaisesRegex(RuntimeError, "segment 1/1.*Argos inference failed"):
            translate_segments([{"text": "source"}], "fr", src_lang="en", mode="argos")

    def test_model_selection_preserves_canonical_repository_case(self):
        self.write_module("torch.py", "device = str")
        self.write_module("transformers.py", """
            class AutoTokenizer:
                @classmethod
                def from_pretrained(cls, model_name, **options):
                    raise RuntimeError('requested ' + model_name)

            AutoModelForSeq2SeqLM = AutoTokenizer
        """)
        for mode, repo in [("m2m100", "facebook/m2m100_418M"),
                           ("m2m100_418M", "facebook/m2m100_418M"),
                           ("m2m100_1.2B", "facebook/m2m100_1.2B"),
                           ("model:MyOrg/Custom", "MyOrg/Custom")]:
            with self.subTest(mode=mode):
                with self.assertRaises(RuntimeError) as exc:
                    translate_segments([{"text": "text"}], "fr", src_lang="en", mode=mode)
                self.assertIn("requested " + repo, str(exc.exception))


if __name__ == "__main__":
    unittest.main()
