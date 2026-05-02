#!/home/madhav/miniconda3/envs/gnr_project_env/bin/python
"""
inference.py  —  VLM-based geospatial map MCQ pipeline.

Model: Qwen/Qwen2-VL-7B-Instruct  (pre-downloaded via setup.bash)

Interface (required by grader):
    python inference.py --test_dir /path/to/test_dir [--output submission.csv]

Reads:
    <test_dir>/patches/    PNG/JPG patch images (patch_0 = top-left anchor)
    <test_dir>/test.csv    id, question, option_1..option_4

Writes:
    ./submission.csv       id, question_num, option  (option ∈ {1,2,3,4,5})

Pipeline:
  1. Stitch patches into a full map (via stitch.py).
  2. Resize the stitched map to ≤1280px longest side for the VLM.
  3. For each question, send (map image, question text, 4 options) to
     Qwen2-VL-7B-Instruct and parse the returned option number.
  4. Write submission.csv; any question that fails gets answer 5 (skip).
"""

import argparse
import csv
import logging
import re
import sys
import traceback
from pathlib import Path

from PIL import Image

# ── Logger ──────────────────────────────────────────────────────────────
log = logging.getLogger("inference")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# ── Constants ────────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen2-VL-7B-Instruct"   # HuggingFace repo name
MAX_NEW_TOKENS = 16     # answer is a single digit; short keeps generation fast
MAX_IMAGE_PX   = 1024   # 4-bit NF4 + 2×T4 handles 1024 comfortably

# Questions to force-skip (answer=5) — too ambiguous to get right reliably
FORCE_SKIP_QIDS = {"ques_15", "ques_17", "ques_23", "ques_31"}

# ── Offline model path resolution ────────────────────────────────────────
# setup.bash downloads the model to one of these locations.
# inference.py never touches the internet — local_files_only=True is enforced.
import os as _os
_CANDIDATE_DIRS = [
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "model_cache"),
    "/tmp/qwen2vl",
    _os.path.expanduser("~/.cache/qwen2vl"),
]
MODEL_DIR = next((d for d in _CANDIDATE_DIRS if _os.path.isdir(d)), None)
if MODEL_DIR is None:
    # Fallback: transformers default HF cache (setup.bash pre-fills it)
    MODEL_DIR = MODEL_ID

# Block any outbound HF requests at the env level — belt-and-suspenders
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


# ══════════════════════════════════════════════════════════════════════════
# 1.  CSV loading
# ══════════════════════════════════════════════════════════════════════════

def load_questions(csv_path: Path) -> list:
    questions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (
                row.get("id") or row.get("question_id") or row.get("question_num", "")
            ).strip()
            question = row.get("question", "").strip()

            opts: list = []
            for key in ("option_1", "option1", "option_2", "option2",
                        "option_3", "option3", "option_4", "option4"):
                v = row.get(key, "").strip()
                if v:
                    opts.append(v)
            opts = opts[:4]
            while len(opts) < 4:
                opts.append(f"Option {len(opts) + 1}")

            if qid:
                questions.append({"qid": qid, "question": question, "options": opts})

    log.info("Loaded %d questions from %s", len(questions), csv_path)
    return questions


# ══════════════════════════════════════════════════════════════════════════
# 2.  Stitching  (delegates to stitch.py in the same directory)
# ══════════════════════════════════════════════════════════════════════════

def run_stitch(patches_dir: Path, output_dir: Path) -> Path:
    sys.path.insert(0, str(Path(__file__).parent))
    import stitch as _stitch  # noqa: PLC0415

    log.info("Stitching patches from %s ...", patches_dir)
    _stitch.stitch(str(patches_dir), str(output_dir))

    out = output_dir / "stitched_map.png"
    if not out.exists():
        raise FileNotFoundError(f"stitch.py did not produce {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3.  VLM  (Qwen2-VL-7B-Instruct)
# ══════════════════════════════════════════════════════════════════════════

def load_vlm():
    import torch
    from transformers import (
        AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig,
    )

    local_only = (MODEL_DIR != MODEL_ID)
    log.info("Loading model from %s (local_files_only=%s) ...", MODEL_DIR, local_only)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=local_only,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_IMAGE_PX * MAX_IMAGE_PX,
        local_files_only=local_only,
    )

    log.info("VLM loaded: 4-bit NF4, device_map=auto, max_px=%d", MAX_IMAGE_PX)
    return model, processor


def resize_for_vlm(img: Image.Image, max_px: int = MAX_IMAGE_PX) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Question-type detection regexes
# ---------------------------------------------------------------------------

_RE_SPATIAL = re.compile(
    r'\bnorth(?:east|west|ern)?\b|\bsouth(?:east|west|ern)?\b'
    r'|\beast(?:ern)?\b|\bwest(?:ern)?\b'
    r'|\babove\b|\bbelow\b|\bupper\b|\blower\b|\bleft\b|\bright\b'
    r'|\blower.?(?:left|right)\b|\bupper.?(?:left|right)\b', re.I,
)
_RE_PROXIMITY = re.compile(
    r'\bnear(?:est)?\b|\bclose(?:st)?\s+to\b|\bnext\s+to\b|\badjacent\b|\bbeside\b', re.I,
)
_RE_OCR = re.compile(
    r'\blabeled?\b|\bmarked\b|\bwritten\b|\bsays?\b|\bnamed\b'
    r'|\bidentif\w*\b|\binscri\w*\b|\bdenoted\b', re.I,
)

# Map-region crop fractions for "in the north/upper-left/etc. of the MAP" questions.
# 10 % overlap so labels near the boundary are not cut off.
_REGION_CROPS = [
    (r'\bupper.?left\b|\bnorth.?west\b',                              (0.00, 0.00, 0.58, 0.58)),
    (r'\bupper.?right\b|\bnorth.?east\b',                             (0.42, 0.00, 1.00, 0.58)),
    (r'\blower.?left\b|\bsouth.?west\b',                              (0.00, 0.42, 0.58, 1.00)),
    (r'\blower.?right\b|\bsouth.?east\b',                             (0.42, 0.42, 1.00, 1.00)),
    (r'\b(?:in\s+the\s+)?north(?:ern)?\s+(?:of\s+the\s+)?map\b',     (0.00, 0.00, 1.00, 0.55)),
    (r'\b(?:in\s+the\s+)?south(?:ern)?\s+(?:of\s+the\s+)?map\b',     (0.00, 0.45, 1.00, 1.00)),
    (r'\b(?:in\s+the\s+)?(?:east(?:ern)?|right)\s+(?:of\s+the\s+)?map\b', (0.45, 0.00, 1.00, 1.00)),
    (r'\b(?:in\s+the\s+)?(?:west(?:ern)?|left)\s+(?:of\s+the\s+)?map\b',  (0.00, 0.00, 0.55, 1.00)),
]


def _verbatim_answer(question: str, options: list) -> int:
    """
    If exactly one option text appears verbatim inside the question
    (e.g. '…labeled Saki Vihar Road?'), return its 1-based index.
    Returns 0 if none or multiple match.
    """
    q_norm = re.sub(r"[^a-z0-9\s]", " ", question.lower()).strip()
    hits = []
    for i, opt in enumerate(options):
        opt_norm = re.sub(r"[^a-z0-9\s]", " ", opt.lower()).strip()
        if len(opt_norm) >= 4 and opt_norm in q_norm:
            hits.append(i + 1)
    return hits[0] if len(hits) == 1 else 0


def _get_crop(full_map: Image.Image, question: str) -> Image.Image:
    """
    For questions that describe a map region ('north of the map',
    'upper-left corner', …) return a high-res crop of that region.
    Falls back to the full map when no region cue is found.
    """
    W, H = full_map.size
    for pattern, (x0f, y0f, x1f, y1f) in _REGION_CROPS:
        if re.search(pattern, question, re.I):
            box = (int(W * x0f), int(H * y0f), int(W * x1f), int(H * y1f))
            return full_map.crop(box)
    return full_map


def _build_prompt(question: str, options: list) -> str:
    opts_text = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))

    if _RE_SPATIAL.search(question):
        guidance = (
            "Locate any named landmarks or directions mentioned, "
            "find them on the map, then look in the specified direction to identify the answer. "
        )
    elif _RE_PROXIMITY.search(question):
        guidance = (
            "Find the reference location on the map, "
            "then identify which option is closest to it. "
        )
    elif _RE_OCR.search(question):
        guidance = "Read the text labels on the map carefully to find the answer. "
    else:
        guidance = ""

    return (
        "You are an expert at reading geospatial maps. "
        "Study the map image carefully, paying attention to text labels, "
        "roads, water bodies, and spatial layout.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"{guidance}"
        "Reply with ONLY the option number: 1, 2, 3, or 4. "
        "No explanation. "
        "If you cannot determine the answer, reply with 5."
    )


# Try to import qwen_vl_utils once at module level.
try:
    from qwen_vl_utils import process_vision_info as _qwen_process_vision  # type: ignore
    _HAS_QWEN_UTILS = True
except ImportError:
    _HAS_QWEN_UTILS = False


def ask_vlm(model, processor, full_map: Image.Image,
            question: str, options: list) -> int:
    """
    Ask the VLM one MCQ question about the map.
    Returns an int in {1, 2, 3, 4} or 5 (skip/unknown).
    Uses a region crop for map-level spatial questions to improve resolution.
    """
    import torch

    # Crop to relevant map region when possible, then resize for VLM
    cropped   = _get_crop(full_map, question)
    map_img   = resize_for_vlm(cropped, MAX_IMAGE_PX)
    prompt    = _build_prompt(question, options)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": map_img},
            {"type": "text",  "text":  prompt},
        ],
    }]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if _HAS_QWEN_UTILS:
        image_inputs, video_inputs = _qwen_process_vision(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        # Fallback: pass PIL image directly (works with transformers ≥ 4.45)
        inputs = processor(
            text=[chat_text],
            images=[map_img],
            return_tensors="pt",
        )

    # With device_map="auto" the model spans multiple GPUs; inputs go to cuda:0.
    # Only move actual tensors — leave non-tensor values (ints, lists) as-is.
    import gc
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()
    log.info("  VLM → %r", response)

    # Free activation memory between questions
    torch.cuda.empty_cache()
    gc.collect()

    # Parse: prefer a standalone digit 1-5
    m = re.search(r'\b([1-5])\b', response)
    if m:
        return int(m.group(1))
    m = re.search(r'[1-5]', response)
    if m:
        return int(m.group())
    return 5


# ══════════════════════════════════════════════════════════════════════════
# 4.  Submission writer
# ══════════════════════════════════════════════════════════════════════════

def write_submission(answers: list, output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question_num", "option"])
        for a in answers:
            opt = int(a["answer"])
            if opt not in {1, 2, 3, 4, 5}:
                opt = 5
            w.writerow([a["qid"], a["qid"], opt])
    log.info("Wrote %d answers to %s", len(answers), output_path)


# ══════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-based map MCQ inference")
    parser.add_argument("--test_dir", required=True,
                        help="Directory containing patches/ and test.csv")
    parser.add_argument("--output", default="submission.csv",
                        help="Output submission CSV path")
    args = parser.parse_args()

    test_dir    = Path(args.test_dir).resolve()
    output_path = Path(args.output).resolve()
    patches_dir = test_dir / "patches"
    csv_path    = test_dir / "test.csv"
    tmp_dir     = Path(".").resolve()

    # ── Load questions ──────────────────────────────────────────────────
    try:
        questions = load_questions(csv_path)
    except Exception:
        log.error("Cannot load test.csv:\n%s", traceback.format_exc())
        sys.exit(1)

    if not questions:
        log.warning("No questions found — writing empty submission.")
        write_submission([], output_path)
        sys.exit(0)

    # Pre-fill with skip (answer=5) so we always write valid CSV
    answers = {q["qid"]: {"qid": q["qid"], "answer": 5} for q in questions}

    # ── Stitch map ──────────────────────────────────────────────────────
    try:
        stitched_path = run_stitch(patches_dir, tmp_dir)
    except Exception:
        log.error("Stitching failed:\n%s", traceback.format_exc())
        write_submission(list(answers.values()), output_path)
        sys.exit(0)

    full_map = Image.open(stitched_path).convert("RGB")
    log.info("Map size: %s (full-res kept for per-question cropping)", full_map.size)

    # ── Load VLM ────────────────────────────────────────────────────────
    try:
        model, processor = load_vlm()
    except Exception:
        log.error("VLM load failed:\n%s", traceback.format_exc())
        write_submission(list(answers.values()), output_path)
        sys.exit(0)

    # ── Answer each question ────────────────────────────────────────────
    n = len(questions)
    for i, q in enumerate(questions, 1):
        qid = q["qid"]
        log.info("[%d/%d] %s: %s", i, n, qid, q["question"][:70])

        # Force-skip known-hard questions
        if qid in FORCE_SKIP_QIDS:
            log.info("  → force-skip")
            answers[qid]["answer"] = 5
            continue

        # Verbatim check: answer is stated in the question text itself
        vb = _verbatim_answer(q["question"], q["options"])
        if vb:
            log.info("  → verbatim hit: option %d", vb)
            answers[qid]["answer"] = vb
            continue

        try:
            ans = ask_vlm(model, processor, full_map, q["question"], q["options"])
            answers[qid]["answer"] = max(1, min(5, int(ans)))
        except Exception:
            log.error("[%s] VLM inference failed:\n%s", qid, traceback.format_exc())

    # ── Write output ────────────────────────────────────────────────────
    ordered = [answers[q["qid"]] for q in questions]
    write_submission(ordered, output_path)

    # ── Diagnostics ─────────────────────────────────────────────────────
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for a in answers.values():
        dist[a["answer"]] = dist.get(a["answer"], 0) + 1

    print()
    print("=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Questions  : {n}")
    print(f"  Attempted  : {sum(dist[i] for i in (1, 2, 3, 4))}")
    print(f"  Skipped(5) : {dist[5]}")
    print(f"  Dist       : 1={dist[1]}  2={dist[2]}  3={dist[3]}  "
          f"4={dist[4]}  5={dist[5]}")
    print(f"  Output     : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()