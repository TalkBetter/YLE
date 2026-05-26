#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


# ===== 在這裡改設定 =====
DATA_JS = PROJECT_DIR / "data.js"
TRANSCRIPT_DIR = PROJECT_DIR / "MOVERS_DATA_TTS_transcripts"
AUDIO_EXTENSIONS = (".wav", ".mp3")

# False: 只處理 INPUT_AUDIO_FILES
# True: 處理 TRANSCRIPT_DIR 裡全部 .wav / .mp3
PROCESS_ALL_AUDIO = False

# 可以寫完整路徑，也可以只寫檔名，例如 "03_body_health.wav" 或 "03_body_health.mp3"
INPUT_AUDIO_FILES = [
    "04_adjectives_feelings.wav",
]

# 空字串代表從檔名自動判斷，例如 03_body_health.wav/mp3 -> body_health
CATEGORY_ID = ""

# "auto": transcript 比單字多 1 行時，切出分類介紹音檔
# True: 一律切分類介紹音檔
# False: 只切 words 裡的句子
INCLUDE_CATEGORY_AUDIO = "auto"

# 測試先輸出到這裡；要正式覆蓋 sentence 時改成 PROJECT_DIR / "sentence"
SENTENCE_RANGE: tuple[int, int] | None = None
OUTPUT_DIR = TRANSCRIPT_DIR / "split_test_04_adjectives_feelings"

FFMPEG = PROJECT_DIR / "ffmpeg.exe"
FFPROBE = PROJECT_DIR / "ffprobe.exe"

NOISE = "-35dB"
SILENCE_DURATION = 0.25
SILENCE_DURATION_FALLBACKS = (0.10, 0.05, 0.03)
MERGE_GAP = 0.05
TRAILING_TOLERANCE = 0.05
MP3_QUALITY = "4"
OVERWRITE_EXISTING = True
DRY_RUN = False
# ===== 設定到這裡 =====


@dataclass(frozen=True)
class WordItem:
    en: str
    ex_en: str
    audio: str


@dataclass(frozen=True)
class Category:
    category_id: str
    label: str
    cat_example_en: str
    audio: str
    words: list[WordItem]


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def find_matching(text: str, start: int, opener: str, closer: str) -> int:
    if text[start] != opener:
        raise ValueError(f"Expected {opener!r} at offset {start}")

    depth = 0
    i = start
    quote: str | None = None
    escape = False
    line_comment = False
    block_comment = False

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            i += 1
            continue

        if line_comment:
            if ch in "\r\n":
                line_comment = False
            i += 1
            continue

        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i

        i += 1

    raise ValueError(f"No matching {closer!r} found after offset {start}")


def find_property_value(text: str, key: str) -> int | None:
    match = re.search(r"(?<![\w$])" + re.escape(key) + r"\s*:", text)
    if not match:
        return None
    return skip_ws(text, match.end())


def parse_js_string(text: str, start: int) -> tuple[str, int]:
    quote = text[start]
    if quote not in ("'", '"', "`"):
        raise ValueError(f"Expected a JS string at offset {start}")

    i = start + 1
    escape = False
    while i < len(text):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == quote:
            literal = text[start : i + 1]
            if quote == "`":
                return text[start + 1 : i], i + 1
            try:
                return ast.literal_eval(literal), i + 1
            except (SyntaxError, ValueError):
                raw = text[start + 1 : i]
                raw = raw.replace(r"\/", "/")
                raw = raw.replace(r"\'", "'").replace(r'\"', '"')
                raw = raw.replace(r"\n", "\n").replace(r"\r", "\r")
                raw = raw.replace(r"\t", "\t").replace(r"\\", "\\")
                return raw, i + 1
        i += 1

    raise ValueError(f"Unterminated JS string at offset {start}")


def extract_string_property(text: str, key: str, default: str = "") -> str:
    value_start = find_property_value(text, key)
    if value_start is None:
        return default
    value_start = skip_ws(text, value_start)
    if value_start >= len(text) or text[value_start] not in ("'", '"', "`"):
        return default
    value, _ = parse_js_string(text, value_start)
    return value


def extract_array_property(text: str, key: str) -> str:
    value_start = find_property_value(text, key)
    if value_start is None:
        raise ValueError(f"Could not find {key!r} array")
    value_start = skip_ws(text, value_start)
    if value_start >= len(text) or text[value_start] != "[":
        raise ValueError(f"Property {key!r} is not an array")
    end = find_matching(text, value_start, "[", "]")
    return text[value_start : end + 1]


def iter_top_level_objects(array_text: str) -> list[str]:
    if not array_text.startswith("[") or not array_text.endswith("]"):
        raise ValueError("Expected array text")

    body = array_text[1:-1]
    objects: list[str] = []
    i = 0
    quote: str | None = None
    escape = False
    line_comment = False
    block_comment = False

    while i < len(body):
        ch = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""

        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            i += 1
            continue

        if line_comment:
            if ch in "\r\n":
                line_comment = False
            i += 1
            continue

        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue

        if ch == "{":
            end = find_matching(body, i, "{", "}")
            objects.append(body[i : end + 1])
            i = end + 1
            continue

        i += 1

    return objects


def extract_assignment_object(js_text: str, assignment_name: str) -> str:
    match = re.search(re.escape(assignment_name) + r"\s*=", js_text)
    if not match:
        raise ValueError(f"Could not find assignment {assignment_name}")

    start = js_text.find("{", match.end())
    if start == -1:
        raise ValueError(f"Could not find object for {assignment_name}")

    end = find_matching(js_text, start, "{", "}")
    return js_text[start : end + 1]


def load_movers_categories(data_js: Path) -> dict[str, Category]:
    js_text = data_js.read_text(encoding="utf-8-sig")
    movers_object = extract_assignment_object(js_text, "window.MOVERS_DATA")
    categories_array = extract_array_property(movers_object, "categories")

    categories: dict[str, Category] = {}
    for category_text in iter_top_level_objects(categories_array):
        category_id = extract_string_property(category_text, "id")
        if not category_id:
            continue
        label = extract_string_property(category_text, "label")
        cat_example_en = extract_string_property(category_text, "catExampleEn")
        category_audio = extract_string_property(category_text, "audio")
        words_array = extract_array_property(category_text, "words")

        words: list[WordItem] = []
        for word_text in iter_top_level_objects(words_array):
            audio = extract_string_property(word_text, "audio")
            ex_en = extract_string_property(word_text, "exEn")
            en = extract_string_property(word_text, "en")
            if audio and ex_en:
                words.append(WordItem(en=en, ex_en=ex_en, audio=audio))

        categories[category_id] = Category(
            category_id=category_id,
            label=label,
            cat_example_en=cat_example_en,
            audio=category_audio,
            words=words,
        )

    return categories


def infer_category_id(input_audio: Path) -> str:
    extensions = "|".join(re.escape(ext.lstrip(".")) for ext in AUDIO_EXTENSIONS)
    match = re.match(rf"^\d+_(.+?)\.(?:{extensions})$", input_audio.name, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot infer category from {input_audio.name!r}; set CATEGORY_ID"
        )
    return match.group(1)


def run_capture(command: list[str]) -> str:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        command_text = " ".join(command)
        raise RuntimeError(f"Command failed ({result.returncode}): {command_text}\n{result.stdout}")
    return result.stdout


def probe_duration(ffprobe: str, input_audio: Path) -> float:
    output = run_capture(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_audio),
        ]
    )
    return float(output.strip())


def detect_silences(
    ffmpeg: str,
    input_audio: Path,
    noise: str,
    silence_duration: float,
) -> list[Silence]:
    output = run_capture(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(input_audio),
            "-af",
            f"silencedetect=noise={noise}:d={silence_duration:g}",
            "-f",
            "null",
            "NUL" if sys.platform.startswith("win") else "-",
        ]
    )

    silences: list[Silence] = []
    pending_start: float | None = None

    for line in output.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue

        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and pending_start is not None:
            silences.append(Silence(pending_start, float(end_match.group(1))))
            pending_start = None

    return silences


def merge_silences(silences: list[Silence], max_gap: float) -> list[Silence]:
    if not silences:
        return []

    merged: list[Silence] = [silences[0]]
    for silence in silences[1:]:
        last = merged[-1]
        if silence.start <= last.end + max_gap:
            merged[-1] = Silence(last.start, max(last.end, silence.end))
        else:
            merged.append(silence)

    return merged


def make_segment_ranges(
    silences: list[Silence],
    audio_duration: float,
    sentence_count: int,
    trailing_tolerance: float,
) -> list[tuple[float, float]]:
    needed_boundaries = sentence_count - 1
    candidates = [
        silence
        for silence in silences
        if silence.start > 0.05 and silence.end < audio_duration - trailing_tolerance
    ]

    if len(candidates) < needed_boundaries:
        raise ValueError(
            f"Only found {len(candidates)} internal silences; need {needed_boundaries}. "
            "Try lowering NOISE, lowering SILENCE_DURATION, or checking the source audio."
        )

    if len(candidates) > needed_boundaries:
        candidates = sorted(
            sorted(candidates, key=lambda silence: silence.duration, reverse=True)[
                :needed_boundaries
            ],
            key=lambda silence: silence.start,
        )

    cuts = [0.0] + [silence.midpoint for silence in candidates] + [audio_duration]
    return [(cuts[i], cuts[i + 1]) for i in range(sentence_count)]


def read_transcript_lines(transcript_path: Path) -> list[str]:
    if not transcript_path.exists():
        return []
    return [
        line.strip()
        for line in transcript_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip()
    ]


def build_audio_items(
    category: Category,
    transcript_lines: list[str],
    include_category_audio: bool | str,
) -> list[WordItem]:
    include = False
    if isinstance(include_category_audio, str):
        if include_category_audio.lower() == "auto":
            include = (
                bool(category.audio)
                and bool(category.cat_example_en)
                and len(transcript_lines) == len(category.words) + 1
            )
        else:
            include = include_category_audio.lower() in ("1", "true", "yes", "y")
    else:
        include = include_category_audio

    if not include:
        return category.words

    if not category.audio or not category.cat_example_en:
        raise ValueError(f"Category {category.category_id!r} has no category audio/example.")

    return [
        WordItem(en=category.category_id, ex_en=category.cat_example_en, audio=category.audio),
        *category.words,
    ]


def apply_sentence_range(
    items: list,
    sentence_range: tuple[int, int] | None,
    label: str,
) -> list:
    if sentence_range is None:
        return items

    start, end = sentence_range
    if start < 1 or end < start:
        raise ValueError(
            f"Invalid SENTENCE_RANGE {sentence_range!r} for {label}; "
            "expected 1-based inclusive (start, end)."
        )

    selected = items[start - 1 : end]
    if not selected:
        raise ValueError(
            f"SENTENCE_RANGE {sentence_range!r} selects no items from {label} "
            f"(available: 1-{len(items)})."
        )
    return selected


def split_audio(
    ffmpeg: str,
    input_audio: Path,
    output_path: Path,
    start: float,
    end: float,
    quality: str,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_audio),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{end - start:.3f}",
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        quality,
        str(output_path),
    ]
    run_capture(command)


def find_ranges_with_fallbacks(
    ffmpeg: str,
    input_audio: Path,
    noise: str,
    silence_duration: float,
    silence_duration_fallbacks: tuple[float, ...],
    merge_gap: float,
    audio_duration: float,
    sentence_count: int,
    trailing_tolerance: float,
) -> tuple[float, list[Silence], list[tuple[float, float]]]:
    durations = [silence_duration]
    durations.extend(
        duration
        for duration in silence_duration_fallbacks
        if duration not in durations
    )

    last_error: ValueError | None = None
    for duration in durations:
        silences = merge_silences(
            detect_silences(ffmpeg, input_audio, noise, duration),
            merge_gap,
        )
        try:
            ranges = make_segment_ranges(
                silences,
                audio_duration,
                sentence_count,
                trailing_tolerance,
            )
        except ValueError as exc:
            last_error = exc
            continue

        return duration, silences, ranges

    if last_error is not None:
        raise last_error
    raise ValueError("No silence duration values configured.")


def process_one(
    input_audio: Path,
    category: Category,
    transcript_dir: Path,
    output_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    noise: str,
    silence_duration: float,
    silence_duration_fallbacks: tuple[float, ...],
    merge_gap: float,
    trailing_tolerance: float,
    quality: str,
    overwrite: bool,
    dry_run: bool,
    include_category_audio: bool | str,
    sentence_range: tuple[int, int] | None,
) -> None:
    if not input_audio.exists():
        raise FileNotFoundError(input_audio)

    transcript_path = transcript_dir / f"{input_audio.stem}.txt"
    transcript_lines = read_transcript_lines(transcript_path)
    if transcript_lines:
        transcript_lines = apply_sentence_range(
            transcript_lines,
            sentence_range,
            transcript_path.name,
        )
    audio_items = build_audio_items(category, transcript_lines, include_category_audio)
    audio_items = apply_sentence_range(audio_items, sentence_range, input_audio.name)
    sentence_count = len(audio_items)
    if transcript_lines and len(transcript_lines) != sentence_count:
        print(
            f"WARNING: {transcript_path.name} has {len(transcript_lines)} lines, "
            f"but data.js has {sentence_count} audio items for {category.category_id}.",
            file=sys.stderr,
        )

    audio_duration = probe_duration(ffprobe, input_audio)
    used_silence_duration, silences, ranges = find_ranges_with_fallbacks(
        ffmpeg=ffmpeg,
        input_audio=input_audio,
        noise=noise,
        silence_duration=silence_duration,
        silence_duration_fallbacks=silence_duration_fallbacks,
        merge_gap=merge_gap,
        audio_duration=audio_duration,
        sentence_count=sentence_count,
        trailing_tolerance=trailing_tolerance,
    )

    output_paths = [output_dir / item.audio for item in audio_items]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite and not dry_run:
        examples = ", ".join(path.name for path in existing[:5])
        more = "" if len(existing) <= 5 else f", ... ({len(existing)} total)"
        raise FileExistsError(
            f"Output files already exist: {examples}{more}. "
            "Set OVERWRITE_EXISTING = True or change OUTPUT_DIR."
        )

    print(
        f"{input_audio.name}: {sentence_count} sentences, "
        f"{len(silences)} merged silences, d={used_silence_duration:g}, "
        f"{audio_duration:.3f}s -> {output_dir}"
    )

    for index, (item, segment_range, output_path) in enumerate(
        zip(audio_items, ranges, output_paths),
        start=1,
    ):
        start, end = segment_range
        print(
            f"{index:02d} {output_path.name} {start:7.3f}-{end:7.3f}s  {item.ex_en}"
        )
        if not dry_run:
            split_audio(
                ffmpeg=ffmpeg,
                input_audio=input_audio,
                output_path=output_path,
                start=start,
                end=end,
                quality=quality,
                overwrite=overwrite,
            )


def resolve_input_audio(item: str | Path, transcript_dir: Path) -> Path:
    path = Path(item)
    if path.is_absolute():
        return path
    return transcript_dir / path


def main() -> int:
    categories = load_movers_categories(DATA_JS)

    if PROCESS_ALL_AUDIO:
        inputs = sorted(
            path
            for path in TRANSCRIPT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
    else:
        inputs = [resolve_input_audio(item, TRANSCRIPT_DIR) for item in INPUT_AUDIO_FILES]

    if not inputs:
        print("No input audio files found.", file=sys.stderr)
        return 1

    for input_audio in inputs:
        category_id = CATEGORY_ID or infer_category_id(input_audio)
        category = categories.get(category_id)
        if category is None:
            available = ", ".join(sorted(categories))
            raise KeyError(f"Unknown MOVERS category {category_id!r}. Available: {available}")

        process_one(
            input_audio=input_audio,
            category=category,
            transcript_dir=TRANSCRIPT_DIR,
            output_dir=OUTPUT_DIR,
            ffmpeg=str(FFMPEG),
            ffprobe=str(FFPROBE),
            noise=NOISE,
            silence_duration=SILENCE_DURATION,
            silence_duration_fallbacks=SILENCE_DURATION_FALLBACKS,
            merge_gap=MERGE_GAP,
            trailing_tolerance=TRAILING_TOLERANCE,
            quality=MP3_QUALITY,
            overwrite=OVERWRITE_EXISTING,
            dry_run=DRY_RUN,
            include_category_audio=INCLUDE_CATEGORY_AUDIO,
            sentence_range=SENTENCE_RANGE,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
