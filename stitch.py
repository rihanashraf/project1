"""
stitch.py — Geospatial patch stitching module (rewritten).

Algorithm (overlap-aware, rotation-aware, vectorised):
  1. Load patches.
  2. Estimate overlap width O by sweeping candidates and counting near-perfect
     edge NCC matches between patch_0 and a sample of others (across all 4
     rotations and all 4 sides).
  3. Pre-compute z-score normalised, length-1 edge strips for every
     (patch, rotation, side) so NCC reduces to a vectorised dot product.
  4. Greedy BFS placement starting from patch_0 (anchor, rot 0, top-left).
     For each occupied cell × each direction, find the best (unplaced patch,
     rotation) by NCC of the corresponding edge-strip pair. Place if score
     above threshold; rejected candidates can still be picked at another cell
     later in the BFS.
  5. Optional relaxed second pass for any remaining unplaced patches
     (lower threshold, marked WEAK).
  6. Compose canvas using rotated patches at their assigned (col, row, rot).
  7. Save:
       - stitched_map.png
       - stitched_map_annotated.png  (patch_id / row / col / rotation / conf)
       - placement_manifest.json
       - stitch_debug.log
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
log = logging.getLogger("stitch")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
ROTATIONS = [0, 90, 180, 270]
DIRECTIONS = ["R", "D", "L", "U"]
DIR_OFFSET: Dict[str, Tuple[int, int]] = {
    "R": (1, 0), "D": (0, 1), "L": (-1, 0), "U": (0, -1),
}
OPP = {"R": "L", "L": "R", "U": "D", "D": "U"}
SIDE_IDX = {"L": 0, "R": 1, "U": 2, "D": 3}

# Confidence buckets
CONF_CONFIRMED = 0.85
CONF_USABLE = 0.65
CONF_WEAK = 0.45

# Acceptance thresholds for greedy placement
THRESH_PRIMARY = 0.70
THRESH_RELAXED = 0.55

# Margin (best - second) needed without forcing
MARGIN_OK = 0.02


# ---------------------------------------------------------------------------
@dataclass
class PatchInfo:
    patch_id: str
    img: np.ndarray         # float32 RGB (H, W, 3) in [0,1]
    is_low_texture: bool = False


@dataclass
class PlacementRecord:
    patch_id: str
    col: int
    row: int
    rot: int
    conf: float
    status: str             # CONFIRMED | USABLE | WEAK | UNPLACED


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_patches(patches_dir: str) -> Dict[str, PatchInfo]:
    p = Path(patches_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Patches directory not found: {patches_dir}")
    pngs = sorted(p.glob("*.png"))
    if not pngs:
        raise ValueError(f"No PNG files in {patches_dir}")
    out: Dict[str, PatchInfo] = {}
    for fp in pngs:
        img = np.array(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
        out[fp.stem] = PatchInfo(patch_id=fp.stem, img=img)
    if "patch_0" not in out:
        raise ValueError("patch_0.png missing — cannot anchor.")
    return out


def validate_patches(patches: Dict[str, PatchInfo]) -> Tuple[int, int]:
    shapes = {pid: p.img.shape[:2] for pid, p in patches.items()}
    uniq = set(shapes.values())
    if len(uniq) > 1:
        raise ValueError(f"Inconsistent patch sizes: {shapes}")
    H, W = next(iter(uniq))
    return H, W


# ---------------------------------------------------------------------------
# Edge strip helpers
# ---------------------------------------------------------------------------
def _rot(img: np.ndarray, k_deg: int) -> np.ndarray:
    if k_deg == 0:
        return img
    return np.rot90(img, k=k_deg // 90)


def _strip(img: np.ndarray, side: str, O: int) -> np.ndarray:
    H, W = img.shape[:2]
    if side == "L":
        return img[:, :O]
    if side == "R":
        return img[:, W - O:]
    if side == "U":
        return img[:O, :]
    if side == "D":
        return img[H - O:, :]
    raise ValueError(side)


def _ncc_1d(a: np.ndarray, b: np.ndarray) -> float:
    """NCC between two 1-D float arrays."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a - a.mean();          b = b - b.mean()
    sa = a.std(); sb = b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float((a * b).mean() / (sa * sb))


def _strip_ncc_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of per-channel NCC for two RGB strips of equal shape."""
    if a.shape != b.shape:
        return -1.0
    s = 0.0
    for c in range(3):
        s += _ncc_1d(a[..., c].ravel(), b[..., c].ravel())
    return s / 3.0


# ---------------------------------------------------------------------------
# Overlap estimation
# ---------------------------------------------------------------------------
def estimate_overlap_width(patches: Dict[str, PatchInfo], H: int, W: int) -> int:
    """
    Sweep candidate overlap widths and pick the one giving the strongest
    edge-NCC matches between patch_0 and a sample of other patches across
    all rotations and all sides.

    A "good" O is one where we observe near-perfect (≥0.95) matches —
    that signature is exactly how this dataset is generated.
    """
    p0 = patches["patch_0"].img
    others = [pid for pid in patches if pid != "patch_0"]
    if not others:
        return max(4, int(0.10 * min(H, W)))

    # Use ALL other patches: alphabetical sort of "patch_1, patch_10, ..."
    # ordering can otherwise miss the true right/down neighbours when only a
    # prefix is sampled.  4 sides × P × 4 rots × ~10 candidate Os is cheap.
    all_others = others

    min_dim = min(H, W)
    # Candidate overlap widths.  Bias toward "round" values that are common
    # in tile generators (8, 16, 24, 32, 48, 64) plus a sweep for safety.
    candidates = sorted(set(
        list(range(2, min_dim // 2 + 1, 2))
        + [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64,
           int(0.05 * min_dim), int(0.10 * min_dim),
           int(0.15 * min_dim), int(0.20 * min_dim),
           int(0.25 * min_dim), int(0.33 * min_dim), int(0.40 * min_dim)]
    ))
    candidates = [o for o in candidates if 4 <= o <= min_dim // 2]

    best_O = candidates[-1]
    best_metric = -1.0
    for O in candidates:
        per_side_max = []
        per_side_n_perfect = 0
        for side in DIRECTIONS:
            opp = OPP[side]
            sa = _strip(p0, side, O)
            best = -2.0
            for pid in all_others:
                cand_img = patches[pid].img
                for k in range(4):
                    rimg = _rot(cand_img, k * 90)
                    if rimg.shape != p0.shape:
                        continue
                    sb = _strip(rimg, opp, O)
                    sc = _strip_ncc_rgb(sa, sb)
                    if sc > best:
                        best = sc
            per_side_max.append(best)
            if best > 0.95:
                per_side_n_perfect += 1

        # Metric prioritises (a) count of near-perfect sides, then (b) sum,
        # with a mild bias toward larger O so that an accidental near-perfect
        # match at a tiny O does not beat the real overlap.
        metric = (per_side_n_perfect * 1000.0
                  + 10.0 * sum(max(0.0, m) for m in per_side_max)
                  + 0.01 * O)
        log.debug(f"  O={O:3d}  sides={[f'{s:.3f}' for s in per_side_max]}  "
                  f"n_perfect={per_side_n_perfect}  metric={metric:.3f}")
        if metric > best_metric:
            best_metric = metric
            best_O = O

    log.info(f"Overlap estimate: O={best_O}px (metric={best_metric:.3f})")
    return best_O


# ---------------------------------------------------------------------------
# Pre-compute z-score normalised flat edge strips
# ---------------------------------------------------------------------------
def _build_strip_table(
    patches: Dict[str, PatchInfo], O: int, H: int, W: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
        norm: float32 array of shape (P, 4, 4, S)  — normalised flat strip
              indexed by [patch_idx, rotation_idx (0..3), side_idx (LRUD), dim].
              Self-dot of any normalised vector == 1.  Pair dot == NCC of the
              underlying RGB strip (channel-mean averaged).
        ids:  list of patch_ids in order of patch_idx.

    A "normalised" flat strip is computed channel-wise:
        x_c  = strip[..., c].ravel()
        x_c -= mean(x_c)
        x_c /= sqrt(N) * std(x_c)   # zero-mean, unit-second-moment
    Concatenated across c=0,1,2 and divided by sqrt(3) so dot product across
    the full 3·H·O vector equals (1/3) * sum_c NCC_c == channel-averaged NCC.
    """
    ids = sorted(patches.keys(), key=lambda s: (
        0 if s == "patch_0" else 1,
        int(s.split("_")[1]) if s.startswith("patch_") and s.split("_")[1].isdigit() else 1 << 30,
        s,
    ))
    P = len(ids)
    # All strips are H×O (LR sides) or O×W (UD sides); after rotating images
    # they always come back to H==W==... (we require square patches).  If not
    # square the strip lengths still match within a side because we extract
    # from the rotated image which has the same per-rot dims.
    # For a square HxW=128x128 patch and O=32, every strip is shape (128,32,3) → 12288 dims.
    side_dims: Dict[str, int] = {
        "L": H * O * 3, "R": H * O * 3, "U": O * W * 3, "D": O * W * 3,
    }
    # Allocate per-side; we'll keep them separate (can't use a single (4) trailing
    # dim if shapes differ between LR and UD).  For square inputs both match.
    if not (side_dims["L"] == side_dims["U"]):
        # rectangular patches: build a per-side list instead
        norm_per_side: Dict[str, np.ndarray] = {
            sd: np.zeros((P, 4, side_dims[sd]), dtype=np.float32) for sd in DIRECTIONS
        }
        for i, pid in enumerate(ids):
            base = patches[pid].img
            for k_idx in range(4):
                ri = _rot(base, k_idx * 90)
                for sd in DIRECTIONS:
                    s = _strip(ri, sd, O)              # (h, o, 3) or (o, w, 3)
                    flat_channels = []
                    for c in range(3):
                        v = s[..., c].ravel().astype(np.float32)
                        v = v - v.mean()
                        sd_v = v.std()
                        if sd_v < 1e-9:
                            v = np.zeros_like(v)
                        else:
                            v = v / (sd_v * np.sqrt(len(v)))
                        flat_channels.append(v)
                    flat = np.concatenate(flat_channels) / np.sqrt(3.0)
                    norm_per_side[sd][i, k_idx] = flat
        # Pack into a single (P, 4, 4, S) only if all dims equal — else return dict via attribute
        return norm_per_side, ids                    # type: ignore[return-value]

    # Square case
    S = side_dims["L"]
    norm = np.zeros((P, 4, 4, S), dtype=np.float32)
    for i, pid in enumerate(ids):
        base = patches[pid].img
        for k_idx in range(4):
            ri = _rot(base, k_idx * 90)
            for sd in DIRECTIONS:
                s = _strip(ri, sd, O)
                flat_channels = []
                for c in range(3):
                    v = s[..., c].ravel().astype(np.float32)
                    v = v - v.mean()
                    sd_v = v.std()
                    if sd_v < 1e-9:
                        v = np.zeros_like(v)
                    else:
                        v = v / (sd_v * np.sqrt(len(v)))
                    flat_channels.append(v)
                flat = np.concatenate(flat_channels) / np.sqrt(3.0)
                norm[i, k_idx, SIDE_IDX[sd]] = flat
    return norm, ids


# ---------------------------------------------------------------------------
# Greedy BFS placement
# ---------------------------------------------------------------------------
def _greedy_place(
    norm: np.ndarray,
    ids: List[str],
    H: int,
    W: int,
    O: int,
    primary_thresh: float = THRESH_PRIMARY,
    relaxed_thresh: float = THRESH_RELAXED,
) -> Tuple[Dict[str, PlacementRecord], Dict[Tuple[int, int], str]]:
    """
    BFS from patch_0. For each frontier cell × direction, find the best
    (unplaced patch, rotation) by NCC against the source strip. Place
    if the score is above threshold. Ties / low-margin candidates are
    accepted greedily — this is a global frontier expansion, not a per-cell
    deferred queue.

    A second relaxed pass attempts any patches still unplaced.
    """
    P = norm.shape[0]
    pid_to_idx = {pid: i for i, pid in enumerate(ids)}
    anchor = pid_to_idx["patch_0"]

    # placed[idx] -> PlacementRecord;   grid[(c, r)] -> idx
    placed: Dict[int, PlacementRecord] = {
        anchor: PlacementRecord("patch_0", 0, 0, 0, 1.0, "CONFIRMED"),
    }
    grid: Dict[Tuple[int, int], int] = {(0, 0): anchor}
    used_rot: Dict[int, int] = {anchor: 0}     # idx -> rot
    placed_set = {anchor}

    # Frontier: idx whose neighbours haven't been examined yet.
    # We re-enqueue freely; a placement query is cheap once vectorised.
    frontier: List[int] = [anchor]

    def best_match(src_idx: int, src_rot: int, src_side: str,
                   thresh: float) -> Optional[Tuple[int, int, float, float]]:
        """Return (best_idx, best_rot_deg, best_score, second_score) or None."""
        opp_idx = SIDE_IDX[OPP[src_side]]
        src_vec = norm[src_idx, src_rot // 90, SIDE_IDX[src_side]]   # (S,)

        # All candidate vectors of opp side, all rotations, shape (P, 4, S)
        cand = norm[:, :, opp_idx, :]                  # (P, 4, S)
        scores = cand @ src_vec                        # (P, 4)

        # Mask out already-placed patches.
        if placed_set:
            mask_idx = np.fromiter(placed_set, dtype=np.int32, count=len(placed_set))
            scores[mask_idx, :] = -10.0

        # Find best
        flat_idx = int(np.argmax(scores))
        best_p, best_r = divmod(flat_idx, 4)
        best_sc = float(scores[best_p, best_r])
        if best_sc < thresh:
            return None
        # second-best (across patches OR rotations of same patch)
        scores2 = scores.copy()
        scores2[best_p, best_r] = -10.0
        flat_idx2 = int(np.argmax(scores2))
        sp2, sr2 = divmod(flat_idx2, 4)
        second_sc = float(scores2[sp2, sr2])
        return best_p, best_r * 90, best_sc, second_sc

    def try_place(thresh: float) -> int:
        """One full sweep: for every placed patch × every empty neighbour,
        try to place. Returns count of patches newly placed."""
        n_new = 0
        # iterate over a snapshot of currently placed cells
        for idx in list(placed.keys()):
            rec = placed[idx]
            for side, (dc, dr) in DIR_OFFSET.items():
                npos = (rec.col + dc, rec.row + dr)
                if npos in grid:
                    continue
                m = best_match(idx, rec.rot, side, thresh)
                if m is None:
                    continue
                bp, brot, bsc, ssc = m
                conf = float(np.clip(bsc, 0.0, 1.0))
                if bsc >= CONF_CONFIRMED:
                    status = "CONFIRMED"
                elif bsc >= CONF_USABLE:
                    status = "USABLE"
                else:
                    status = "WEAK"
                placed[bp] = PlacementRecord(
                    patch_id=ids[bp], col=npos[0], row=npos[1],
                    rot=brot, conf=conf, status=status,
                )
                grid[npos] = bp
                placed_set.add(bp)
                n_new += 1
        return n_new

    # Main loop: keep sweeping until no new placements.
    while True:
        added = try_place(primary_thresh)
        if added == 0:
            break

    # Relaxed pass for any leftover unplaced patches.
    n_unplaced = P - len(placed)
    if n_unplaced > 0:
        log.info(f"Primary pass placed {len(placed)}/{P}; running relaxed pass.")
        while True:
            added = try_place(relaxed_thresh)
            if added == 0:
                break

    # Build patch_id-keyed dict
    out: Dict[str, PlacementRecord] = {}
    for idx, rec in placed.items():
        out[ids[idx]] = rec

    grid_pid: Dict[Tuple[int, int], str] = {pos: ids[idx] for pos, idx in grid.items()}
    return out, grid_pid


# ---------------------------------------------------------------------------
# Canvas composition (rotated patches, average-blend in overlap)
# ---------------------------------------------------------------------------
def compose_canvas(
    patches: Dict[str, PatchInfo],
    placed: Dict[str, PlacementRecord],
    H: int, W: int, O: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    if not placed:
        raise ValueError("No patches placed — cannot compose canvas.")
    cols = [r.col for r in placed.values()]
    rows = [r.row for r in placed.values()]
    minc, maxc = min(cols), max(cols)
    minr, maxr = min(rows), max(rows)
    step = max(W - O, 1)
    cw = (maxc - minc + 1) * step + O
    ch = (maxr - minr + 1) * step + O

    accum = np.zeros((ch, cw, 3), dtype=np.float64)
    weight = np.zeros((ch, cw), dtype=np.float64)

    # Distance-from-edge weight (centre = 1, border ≈ 1e-3)
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float64),
        np.arange(W, dtype=np.float64),
        indexing="ij",
    )
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    dy = 1.0 - np.abs(yy - cy) / max(cy, 1e-6)
    dx = 1.0 - np.abs(xx - cx) / max(cx, 1e-6)
    w_ramp = np.clip(dy * dx, 1e-3, 1.0)

    for rec in placed.values():
        rimg = _rot(patches[rec.patch_id].img, rec.rot)
        ph, pw = rimg.shape[:2]
        px = (rec.col - minc) * step
        py = (rec.row - minr) * step
        x_end = min(px + pw, cw)
        y_end = min(py + ph, ch)
        sx = x_end - px
        sy = y_end - py
        accum[py:y_end, px:x_end] += rimg[:sy, :sx].astype(np.float64) * w_ramp[:sy, :sx, None]
        weight[py:y_end, px:x_end] += w_ramp[:sy, :sx]

    canvas = np.full((ch, cw, 3), 0.5, dtype=np.float32)
    covered = weight > 0.0
    safe = np.where(covered, weight, 1.0)
    blended = (accum / safe[:, :, None]).astype(np.float32)
    canvas[covered] = np.clip(blended[covered], 0.0, 1.0)
    out = (canvas * 255).round().astype(np.uint8)
    return out, (minc, minr)


# ---------------------------------------------------------------------------
# Annotated canvas
# ---------------------------------------------------------------------------
def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def annotate_canvas(
    canvas: np.ndarray,
    placed: Dict[str, PlacementRecord],
    origin: Tuple[int, int],
    H: int, W: int, O: int,
) -> np.ndarray:
    minc, minr = origin
    step = max(W - O, 1)
    pil = Image.fromarray(canvas).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    font_id = _font(11)
    font_meta = _font(9)

    for rec in placed.values():
        px = (rec.col - minc) * step
        py = (rec.row - minr) * step
        # Cell border
        col = (220, 30, 30) if rec.status == "WEAK" else (30, 130, 30)
        draw.rectangle([px, py, px + W - 1, py + H - 1], outline=col, width=1)
        # Label panel (semi-transparent black background for legibility)
        lab1 = rec.patch_id
        lab2 = f"r{rec.row} c{rec.col} rot{rec.rot}"
        lab3 = f"{rec.status} {rec.conf:.2f}"
        # crude bg box
        draw.rectangle([px, py, px + min(W, 86), py + 36], fill=(0, 0, 0, 160))
        draw.text((px + 2, py + 1),  lab1, fill=(255, 255, 255), font=font_id)
        draw.text((px + 2, py + 13), lab2, fill=(220, 220, 0), font=font_meta)
        draw.text((px + 2, py + 23), lab3, fill=(180, 220, 255), font=font_meta)

    return np.array(pil)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def save_manifest(
    patches: Dict[str, PatchInfo],
    placed: Dict[str, PlacementRecord],
    output_dir: str,
) -> List[dict]:
    rows: List[dict] = []
    for pid in sorted(patches.keys(), key=lambda s: (
        int(s.split("_")[1]) if s.startswith("patch_") and s.split("_")[1].isdigit() else 1 << 30,
        s,
    )):
        if pid in placed:
            r = placed[pid]
            rows.append({
                "patch_id": r.patch_id, "col": r.col, "row": r.row,
                "rot": r.rot, "conf": round(r.conf, 6), "status": r.status,
            })
        else:
            rows.append({
                "patch_id": pid, "col": None, "row": None,
                "rot": None, "conf": 0.0, "status": "UNPLACED",
            })
    out = Path(output_dir) / "placement_manifest.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Manifest: {out}  ({len(rows)} entries)")
    return rows


def save_stitched(canvas: np.ndarray, output_dir: str) -> Path:
    out = Path(output_dir) / "stitched_map.png"
    Image.fromarray(canvas).save(out)
    log.info(f"Stitched map: {out}")
    return out


def save_annotated(canvas: np.ndarray, output_dir: str) -> Path:
    out = Path(output_dir) / "stitched_map_annotated.png"
    Image.fromarray(canvas).save(out)
    log.info(f"Annotated map: {out}")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
_last_O: List[int] = [0]


def stitch(patches_dir: str, output_dir: str = ".") -> Dict[str, PlacementRecord]:
    """
    Run the full stitching pipeline.  Always writes:
        stitched_map.png, stitched_map_annotated.png,
        placement_manifest.json, stitch_debug.log
    Returns: placed dict (patch_id → PlacementRecord).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # File handler for stitch_debug.log (overwritten each run)
    fh = logging.FileHandler(Path(output_dir) / "stitch_debug.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%H:%M:%S"))
    log.addHandler(fh)

    try:
        patches = load_patches(patches_dir)
        H, W = validate_patches(patches)
        log.info(f"Loaded {len(patches)} patches; size H={H} W={W}")

        O = estimate_overlap_width(patches, H, W)
        _last_O[0] = O

        log.info("Building edge-strip table…")
        norm_or_dict, ids = _build_strip_table(patches, O, H, W)

        # Vectorised greedy placement only supports the square case;
        # for the rectangular-patch path we fall back to a slower per-side loop.
        if isinstance(norm_or_dict, np.ndarray):
            log.info(f"Strip table: shape={norm_or_dict.shape}, dtype={norm_or_dict.dtype}")
            placed_pid, _grid = _greedy_place(norm_or_dict, ids, H, W, O)
        else:
            log.warning("Rectangular patches detected — using per-side fallback.")
            placed_pid, _grid = _greedy_place_per_side(norm_or_dict, ids, H, W, O)

        n_placed = len(placed_pid)
        coverage = n_placed / len(patches)
        log.info(f"Placement complete: {n_placed}/{len(patches)} ({coverage:.1%})")

        # Compose visuals
        canvas, origin = compose_canvas(patches, placed_pid, H, W, O)
        save_stitched(canvas, output_dir)
        annotated = annotate_canvas(canvas, placed_pid, origin, H, W, O)
        save_annotated(annotated, output_dir)
        save_manifest(patches, placed_pid, output_dir)

        return placed_pid

    finally:
        log.removeHandler(fh)
        try:
            fh.close()
        except Exception:
            pass


def _greedy_place_per_side(
    norm_per_side: Dict[str, np.ndarray],
    ids: List[str],
    H: int, W: int, O: int,
    primary_thresh: float = THRESH_PRIMARY,
    relaxed_thresh: float = THRESH_RELAXED,
) -> Tuple[Dict[str, PlacementRecord], Dict[Tuple[int, int], str]]:
    """Fallback for rectangular patches; semantics match _greedy_place."""
    P = next(iter(norm_per_side.values())).shape[0]
    pid_to_idx = {pid: i for i, pid in enumerate(ids)}
    anchor = pid_to_idx["patch_0"]
    placed: Dict[int, PlacementRecord] = {
        anchor: PlacementRecord("patch_0", 0, 0, 0, 1.0, "CONFIRMED"),
    }
    grid: Dict[Tuple[int, int], int] = {(0, 0): anchor}
    placed_set = {anchor}

    def best_match(src_idx: int, src_rot: int, src_side: str, thresh: float):
        opp = OPP[src_side]
        # source strip and candidate strips have different shapes if rectangular
        # but their flat dims must match (same H*O*3 vs O*W*3).  Skip pairs whose
        # side dim differs from the source's.
        src_vec = norm_per_side[src_side][src_idx, src_rot // 90]
        cand_block = norm_per_side[opp]
        if cand_block.shape[-1] != src_vec.shape[-1]:
            return None
        scores = cand_block @ src_vec  # (P, 4)
        if placed_set:
            mask_idx = np.fromiter(placed_set, dtype=np.int32, count=len(placed_set))
            scores[mask_idx, :] = -10.0
        flat = int(np.argmax(scores))
        bp, br = divmod(flat, 4)
        bs = float(scores[bp, br])
        if bs < thresh:
            return None
        return bp, br * 90, bs

    def sweep(thresh: float) -> int:
        n = 0
        for idx in list(placed.keys()):
            rec = placed[idx]
            for side, (dc, dr) in DIR_OFFSET.items():
                npos = (rec.col + dc, rec.row + dr)
                if npos in grid:
                    continue
                m = best_match(idx, rec.rot, side, thresh)
                if m is None:
                    continue
                bp, brot, bs = m
                if bs >= CONF_CONFIRMED:
                    st = "CONFIRMED"
                elif bs >= CONF_USABLE:
                    st = "USABLE"
                else:
                    st = "WEAK"
                placed[bp] = PlacementRecord(
                    patch_id=ids[bp], col=npos[0], row=npos[1],
                    rot=brot, conf=float(np.clip(bs, 0, 1)), status=st,
                )
                grid[npos] = bp
                placed_set.add(bp)
                n += 1
        return n

    while sweep(primary_thresh):
        pass
    if P - len(placed) > 0:
        while sweep(relaxed_thresh):
            pass

    return ({ids[idx]: rec for idx, rec in placed.items()},
            {pos: ids[idx] for pos, idx in grid.items()})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("patches_dir")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    log.setLevel(getattr(logging, args.log_level))
    placed = stitch(args.patches_dir, args.output_dir)
    n_placed = len(placed)
    print(f"\nDone. {n_placed} patches placed.")
    print(f"Outputs in: {args.output_dir}/")
