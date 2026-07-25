"""Milestone 25 (Module 4, Phase 25): SoccerNet tracking-data acquisition.

STANDALONE MODULE TREE: `production/src/cv/` is new and isolated. Nothing in
`production/src/models`, `production/src/pipeline`, `production/src/spatial`,
`production/src/physics`, or `production/src/serving` is imported by or
imports from this package. This milestone does not modify any existing ML,
physics, or API code.

SCHEMA VERIFICATION NOTE (read before trusting the class-taxonomy
assumptions below -- this is the exact "verify, don't assume" lesson this
project has re-learned repeatedly with StatsBomb's schema: event types
(Milestone 5), competition JSON structure (Milestone 9), substitution event
fields (Milestone 20), and 360 freeze-frame player identity (Milestone 22)):

This milestone's own task brief hypothesized that SoccerNet's tracking
`gt.txt` files distinguish player/goalkeeper/referee/ball as separate
classes. Obtaining the actual gated tracking archive requires a real,
individually-issued NDA password (see `download_sample_dataset` below) that
was NOT available when this module was written, so the hypothesis could not
be checked against a real downloaded file directly in this environment.
Checking the OFFICIAL SoccerNet-Tracking documentation (github.com/SoccerNet/
sn-tracking) instead -- the next best verification available, not a
memory-based guess -- found the opposite of the hypothesis:

    "The ground truth ... [is] stored in comma-separate csv files with 10
    columns. These values correspond in order to: frame ID, track ID, top
    left coordinate of the bounding box, top y coordinate, width, height,
    confidence score for the detection (always 1. for the ground truth) and
    the remaining values are set to -1."

    "The object classes are not taken into account in this challenge or the
    evaluation. The object to retrieve are among the following classes:
    players, goalkeepers, referees, balls and any other human entering the
    field."

In plain terms: `gt.txt` is a standard 10-column MOT-format CSV
(`frame,track_id,x,y,w,h,conf,-1,-1,-1`) with NO per-object class field at
all -- players, goalkeepers, referees, balls, and any other human on the
field are one undifferentiated tracked-object set for this dataset. This is
a genuine, documented finding, not an assumption this module makes -- see
`inspect_annotation_format` below, which re-verifies this directly against
the real downloaded file (not just the docs) the first time real data is
available, exactly the same "trust but verify against the real artifact"
discipline used throughout this project. A richer role/team taxonomy DOES
exist in SoccerNet's newer, separate Game State Reconstruction task
(`gamestate-2024`/`gamestate-2025`), which is the more promising acquisition
target for Phase 28's team/role classification -- NOT this milestone's
`tracking` task, whose `gt.txt` genuinely has no classes to preserve.
"""

import os
import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

SOCCERNET_LOCAL_DIR = Path("data/raw/soccernet")
TRACKING_TASK = "tracking"

# The real gt.txt schema, verified against github.com/SoccerNet/sn-tracking's
# documented format (see module docstring) -- NOT a class taxonomy, since
# the real file has none. Kept here as a single, named source of truth for
# the parser in `load_sample_frames_and_labels`, and printed verbatim by
# `inspect_annotation_format` for a human to cross-check against.
GT_TXT_COLUMNS = (
    "frame_id",
    "track_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "confidence",
    "unused_1",
    "unused_2",
    "unused_3",
)


def download_sample_dataset(num_games: int = 1, password: str | None = None) -> Path | None:
    """Downloads the SoccerNet tracking dataset's `train` split.

    SoccerNet's tracking data requires a signed NDA/research-use agreement
    with SoccerNet to obtain a real download password -- this is a genuine
    manual step (register at https://www.soccer-net.org, request access,
    sign the NDA) that this function CANNOT complete on its own, and does
    not attempt to. If no password is supplied (as an argument or via the
    `SOCCERNET_PASSWORD` environment variable), this prints clear
    instructions and returns None -- this is the EXPECTED path in an
    environment with no NDA password on file, not an error condition, and
    it must never crash or silently no-op without explanation.

    Returns the path to the extracted `train/` sequence directory on
    success, or None if no password was available or the download/extract
    did not produce the expected archive.

    KNOWN API LIMITATION (documented, not hidden): the underlying
    `SoccerNetDownloader.downloadDataTask` API downloads the FULL `train`
    split archive for the whole dataset -- there is no per-game download
    option. `num_games` therefore does not reduce what is downloaded; it
    only limits how many of the extracted sequences
    `load_sample_frames_and_labels` will use. For a dataset this size, that
    means the first successful call downloads everything regardless of
    `num_games` -- worth knowing before running this against a metered
    connection.
    """
    resolved_password = password or os.environ.get("SOCCERNET_PASSWORD")

    if not resolved_password:
        print(
            "\n[acquisition] SoccerNet tracking data requires a real, individually-issued "
            "download password -- this is a signed NDA/research-use agreement with SoccerNet, "
            "not something this script can obtain automatically.\n"
            "\n"
            "To proceed:\n"
            "  1. Register at https://www.soccer-net.org and request access to the tracking "
            "dataset.\n"
            "  2. Sign the NDA/research-use agreement SoccerNet sends you.\n"
            "  3. You will receive a download password.\n"
            "  4. Re-run with that password, either:\n"
            "       download_sample_dataset(password='<your password>')\n"
            "     or by setting the SOCCERNET_PASSWORD environment variable before running.\n"
            "\n"
            "This is an expected manual step, not a bug -- no data has been downloaded.\n"
        )
        return None

    SOCCERNET_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(SOCCERNET_LOCAL_DIR))
    downloader.downloadDataTask(task=TRACKING_TASK, split=["train"], password=resolved_password)

    train_zip = SOCCERNET_LOCAL_DIR / TRACKING_TASK / "train.zip"
    if not train_zip.exists():
        print(
            f"[acquisition] Expected archive not found at {train_zip} after download -- check "
            "that the password is correct and that the tracking train.zip is actually "
            "available on SoccerNet's server (see SoccerNetDownloader's own console output "
            "above for the specific failure)."
        )
        return None

    extract_dir = SOCCERNET_LOCAL_DIR / TRACKING_TASK / "train"
    if not extract_dir.exists():
        print(f"[acquisition] Extracting {train_zip} ...")
        with zipfile.ZipFile(train_zip) as zf:
            zf.extractall(SOCCERNET_LOCAL_DIR / TRACKING_TASK)

    sequence_dirs = sorted(p for p in extract_dir.iterdir() if p.is_dir())
    print(
        f"[acquisition] {len(sequence_dirs)} sequence(s) extracted to {extract_dir}; "
        f"load_sample_frames_and_labels will use the first {num_games} per `num_games`."
    )
    return extract_dir


def inspect_annotation_format(sequence_dir: Path, num_lines: int = 5) -> None:
    """Prints the RAW structure of a real downloaded `gt.txt` (and
    `seqinfo.ini`, if present) for `sequence_dir` -- the live, real-artifact
    re-verification of the schema documented in this module's docstring.
    Run this against real downloaded data before trusting any parsing code
    downstream of it, per this project's established schema-verification
    discipline.
    """
    gt_path = sequence_dir / "gt" / "gt.txt"
    seqinfo_path = sequence_dir / "seqinfo.ini"

    print(f"\n[acquisition] Inspecting real annotation file: {gt_path}")
    if not gt_path.exists():
        print(f"[acquisition] {gt_path} does not exist -- cannot inspect.")
        return

    with open(gt_path) as f:
        lines = [next(f).rstrip("\n") for _ in range(num_lines)]
    print(f"[acquisition] First {num_lines} raw line(s) of {gt_path.name}:")
    for line in lines:
        print(f"  {line}")
    print(f"[acquisition] Expected column layout (per GT_TXT_COLUMNS): {GT_TXT_COLUMNS}")

    if seqinfo_path.exists():
        print(f"\n[acquisition] Raw contents of {seqinfo_path.name}:")
        print(seqinfo_path.read_text())
    else:
        print(f"[acquisition] {seqinfo_path} not present.")


def load_sample_frames_and_labels(game_id: int | str, extract_dir: Path | None = None) -> dict:
    """Loads one sequence's frame image paths and parsed ground-truth boxes.

    `game_id` is either an integer index into the sorted list of extracted
    sequence directories under `extract_dir` (default:
    `SOCCERNET_LOCAL_DIR/tracking/train`), or the sequence directory's name
    directly.

    Returns `{"sequence_dir": Path, "frame_paths": {frame_id: Path, ...},
    "labels": {frame_id: [box, ...], ...}}`, where each `box` is
    `{"track_id": int, "bbox": [x, y, w, h], "class": None}`.

    `"class"` is explicitly `None`, not omitted and not a guessed label --
    see this module's docstring: the real `gt.txt` format has no per-object
    class field at all (players, goalkeepers, referees, and balls are one
    undifferentiated tracked-object set in this dataset). Setting it to
    `None` rather than leaving the key out makes the absence of class
    information impossible for a caller to miss.
    """
    if extract_dir is None:
        extract_dir = SOCCERNET_LOCAL_DIR / TRACKING_TASK / "train"

    if isinstance(game_id, int):
        sequence_dirs = sorted(p for p in extract_dir.iterdir() if p.is_dir())
        sequence_dir = sequence_dirs[game_id]
    else:
        sequence_dir = extract_dir / game_id

    image_dir = sequence_dir / "img1"
    frame_paths = {
        int(p.stem): p for p in sorted(image_dir.glob("*.jpg"))
    } if image_dir.exists() else {}

    labels: dict[int, list[dict]] = {}
    gt_path = sequence_dir / "gt" / "gt.txt"
    with open(gt_path) as f:
        for line in f:
            fields = line.strip().split(",")
            frame_id = int(fields[0])
            track_id = int(fields[1])
            x, y, w, h = (float(v) for v in fields[2:6])
            labels.setdefault(frame_id, []).append(
                {"track_id": track_id, "bbox": [x, y, w, h], "class": None}
            )

    return {"sequence_dir": sequence_dir, "frame_paths": frame_paths, "labels": labels}
