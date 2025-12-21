from typing import List, Dict


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: List[Dict], out_path: str, use_translated: bool = False) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            start = _format_timestamp(seg.get('start', 0.0))
            end = _format_timestamp(seg.get('end', 0.0))
            text = seg.get('translated_text') if use_translated else seg.get('text', '')
            if text is None:
                text = ''
            else:
                text = str(text)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(text.strip() + "\n\n")
