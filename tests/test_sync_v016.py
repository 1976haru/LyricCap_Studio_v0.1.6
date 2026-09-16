from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from lyriccap.aligner import StableTSAligner
from lyriccap.vocal_separator import DemucsSeparator
from lyriccap.vocal_separator import VocalSeparationError


class FakeModel:
    def __init__(self):
        self.calls = []
    def align(self, audio, text, language=None, **opts):
        self.calls.append((audio, text, language, opts))
        lines = text.splitlines()
        segs = []
        t = 1.0
        for line in lines:
            words=[]
            for token in line.split():
                words.append(SimpleNamespace(word=token, start=t, end=t+0.25, probability=0.9))
                t += 0.3
            segs.append(SimpleNamespace(words=words, start=words[0].start, end=words[-1].end))
            t += 0.2
        return SimpleNamespace(segments=segs)

    def transcribe(self, audio, language=None, **opts):
        return self.align(audio, "Hello world\nSing again", language, **opts)


class SyncTests(unittest.TestCase):
    def test_precise_falls_back_to_original_when_demucs_fails(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / 'song.wav'; src.write_bytes(b'RIFF' + b'0' * 100)
            model = FakeModel()
            messages = []
            aligner = StableTSAligner(
                sync_mode='music_precise', cache_dir=td / 'cache',
                fallback_enabled=True, progress=messages.append,
            )
            aligner._model = model
            aligner.separator.separate = lambda _path: (_ for _ in ()).throw(VocalSeparationError('demucs failed'))
            cues = aligner.align(src, ['Hello world'], 'en')
            self.assertEqual(len(cues), 1)
            self.assertEqual(Path(model.calls[0][0]), src)
            self.assertEqual(aligner.last_stats.aligned_audio, 'original-fallback')
            self.assertTrue(any('fallback' in message for message in messages))

    def test_precise_uses_external_vocal_stem_and_no_denoiser_kw(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            src=td/'song.wav'; src.write_bytes(b'RIFF' + b'0'*100)
            vocal=td/'vocals.wav'; vocal.write_bytes(b'RIFF' + b'1'*100)
            model=FakeModel()
            a=StableTSAligner(sync_mode='music_precise', cache_dir=td/'cache')
            a._model=model
            a.separator.separate=lambda _p: vocal
            cues=a.align(src, ['Hello world', 'Sing again'], 'en')
            self.assertEqual(len(cues), 2)
            audio, text, lang, opts=model.calls[0]
            self.assertEqual(Path(audio), vocal)
            self.assertEqual(lang, 'en')
            self.assertNotIn('denoiser', opts)
            self.assertTrue(opts['word_timestamps'])
            self.assertEqual(a.last_stats.aligned_audio, 'demucs-vocals')

    def test_demucs_separator_caches_result(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            src=td/'노래.wav'; src.write_bytes(b'RIFF' + b'0'*100)
            sep=DemucsSeparator(td/'cache')

            def fake_run(cmd, **kwargs):
                out=Path(cmd[cmd.index('-o')+1])
                d=out/'htdemucs'/src.stem
                d.mkdir(parents=True, exist_ok=True)
                (d/'vocals.wav').write_bytes(b'RIFF'+b'2'*100)
                return SimpleNamespace(returncode=0, stdout='ok')

            with patch('lyriccap.vocal_separator.subprocess.run', side_effect=fake_run) as run:
                p1=sep.separate(src)
                p2=sep.separate(src)
                self.assertEqual(p1, p2)
                self.assertTrue(p1.exists())
                self.assertEqual(run.call_count, 1)

if __name__ == '__main__':
    unittest.main()
