from __future__ import annotations

"""
Thousands of Galaxies, One Map — cinematic YouTube Short renderer

Creates a vertical 1080x1920 astronomy short from public Sloan Digital Sky
Survey (SDSS) galaxy spectroscopy. Each plotted point is one galaxy with a
measured sky position and spectroscopic redshift. The animation moves from the
survey footprint on the sky to a redshift-depth wedge and a rotating 3D point
cloud, revealing walls, filaments, clusters, and comparatively empty voids.

Preferred live source
---------------------
SDSS Data Release 16 SkyServer SQL web service. The renderer requests a bounded
sample from SpecObj with:

- class = GALAXY
- zWarning = 0
- 0.01 < redshift < 0.25
- measured right ascension and declination

The catalogue is split into several RA ranges so the result is not dominated by
one database page. The SQL endpoint and query are written into the output
metadata for reproducibility.

Science notes
-------------
- Right ascension and declination give direction on the sky.
- Spectroscopic redshift provides a distance proxy; this script converts z to
  reference comoving distance using a flat Lambda-CDM model with H0=70 km/s/Mpc
  and Omega_m=0.3.
- The wedge map uses a thin declination slice, similar in spirit to classic
  redshift-survey maps.
- Dense ridges and knots trace the cosmic web; sparse regions are voids.
- Sharp footprint boundaries and some gaps are survey geometry and selection,
  not physical edges of the universe.
- This is a visual map, not a precision cosmological analysis.

Offline behaviour
-----------------
If SkyServer is unreachable, the script uses a clearly labelled deterministic
fixture containing clustered knots, filaments, walls, voids, and survey gaps.
The fixture is for preview/layout validation only and is not observational data.

Recommended install
-------------------
    pip install numpy pandas matplotlib pillow imageio imageio-ffmpeg tqdm

Quick preview render
--------------------
    GALAXY_MAP_SHORT_QUICK=1 python thousands_of_galaxies_one_map_short.py

Force offline fixture mode
--------------------------
    GALAXY_MAP_SHORT_OFFLINE=1 python thousands_of_galaxies_one_map_short.py

Use a previously downloaded CSV
-------------------------------
    GALAXY_MAP_DATA_PATH=/path/to/sdss_galaxies.csv \
        python thousands_of_galaxies_one_map_short.py

Primary references
------------------
- SDSS DR16 SkyServer SQL API:
  https://cas.sdss.org/dr16/en/help/docs/api.aspx
- SDSS DR16 SkyServer:
  https://skyserver.sdss.org/dr16/en/home.aspx
- SDSS data releases:
  https://www.sdss.org/science/publications/data-release-publications/
"""

import io
import json
import math
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

QUICK_MODE = os.environ.get("GALAXY_MAP_SHORT_QUICK", "0") == "1"
OFFLINE_MODE = os.environ.get("GALAXY_MAP_SHORT_OFFLINE", "0") == "1"
LOCAL_DATA_PATH = os.environ.get("GALAXY_MAP_DATA_PATH", "").strip()

OUTPUT_ROOT = Path("thousands_of_galaxies_one_map_short_output")
DATA_ROOT = OUTPUT_ROOT / "data"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
for directory in (OUTPUT_ROOT, DATA_ROOT, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SKYSERVER_SQL_URL = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"

CONFIG: Dict[str, Any] = {
    "video_width": 540 if QUICK_MODE else 1080,
    "video_height": 960 if QUICK_MODE else 1920,
    "fps": 6 if QUICK_MODE else 24,
    "duration_s": 12 if QUICK_MODE else 58,
    "output_basename": "thousands_of_galaxies_one_map",
    "title": "THOUSANDS OF GALAXIES, ONE MAP",
    "subtitle": "SDSS spectroscopy // sky position + redshift // cosmic web",
    "data_timeout_s": 35,
    "target_rows": 2400 if QUICK_MODE else 12000,
    "ra_bins": 4 if QUICK_MODE else 8,
    "ra_min": 110.0,
    "ra_max": 270.0,
    "dec_min": -8.0,
    "dec_max": 72.0,
    "z_min": 0.01,
    "z_max": 0.25,
    "reference_h0": 70.0,
    "reference_omega_m": 0.3,
    "max_render_points": 2600 if QUICK_MODE else 10500,
    "background_stars": 280,
    "hud_noise": 48,
    "contrast": 1.08,
    "saturation": 1.06,
    "vignette": 0.24,
}

OUT_W = int(CONFIG["video_width"])
OUT_H = int(CONFIG["video_height"])
OUT_SIZE = (OUT_W, OUT_H)
C_KMS = 299_792.458

COLORS = {
    "ice": (146, 224, 255),
    "cyan": (76, 229, 255),
    "blue": (72, 131, 255),
    "violet": (185, 110, 255),
    "gold": (255, 193, 89),
    "rose": (255, 99, 157),
    "white": (245, 250, 255),
    "muted": (157, 203, 226),
    "dark": (3, 7, 17),
}

FULL_CAPTIONS = [
    (0.5, 7.2, "Every point in this map is a galaxy whose spectrum was measured by the Sloan Digital Sky Survey."),
    (7.3, 17.2, "Right ascension and declination place each galaxy on the sky. The outline is the survey footprint, not an edge of space."),
    (17.3, 27.2, "Colour the same galaxies by redshift and the flat sky map gains depth: nearby systems are blue, more distant systems shift warmer."),
    (27.3, 39.0, "Cut a thin slice through the survey and structure appears—knots, walls, filaments, and broad regions with far fewer galaxies."),
    (39.1, 49.7, "Redshift becomes a reference comoving distance, turning thousands of spectra into a three-dimensional point cloud."),
    (49.8, 57.4, "This is the cosmic web: matter gathered by gravity into connected structure across hundreds of millions of light-years."),
]
if QUICK_MODE:
    _caption_scale = float(CONFIG["duration_s"]) / 58.0
    CAPTIONS = [(a * _caption_scale, b * _caption_scale, text) for a, b, text in FULL_CAPTIONS]
else:
    CAPTIONS = FULL_CAPTIONS

SHOT_PLAN = [
    {"name": "intro", "start": 0.0, "end": 7.8 if not QUICK_MODE else 1.65},
    {"name": "sky_map", "start": 7.8 if not QUICK_MODE else 1.65, "end": 18.5 if not QUICK_MODE else 3.85},
    {"name": "redshift", "start": 18.5 if not QUICK_MODE else 3.85, "end": 29.0 if not QUICK_MODE else 6.05},
    {"name": "wedge", "start": 29.0 if not QUICK_MODE else 6.05, "end": 40.5 if not QUICK_MODE else 8.4},
    {"name": "point_cloud", "start": 40.5 if not QUICK_MODE else 8.4, "end": 50.5 if not QUICK_MODE else 10.45},
    {"name": "finale", "start": 50.5 if not QUICK_MODE else 10.45, "end": float(CONFIG["duration_s"])},
]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def smoothstep(value: float) -> float:
    x = clamp(value)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def get_shot(t: float) -> Dict[str, Any]:
    for shot in SHOT_PLAN:
        if shot["start"] <= t < shot["end"]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> Optional[str]:
    for start, end, text in CAPTIONS:
        if start <= t < end:
            return text
    return None


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    size: int = 28,
    fill=(255, 255, 255, 255),
    bold: bool = False,
    stroke: int = 2,
    anchor: str = "la",
):
    ImageDraw.Draw(image).text(
        xy,
        text,
        font=get_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, min(220, fill[3] if len(fill) > 3 else 220)),
    )


def draw_wrapped_text(
    image: Image.Image,
    text: str,
    xy: Tuple[int, int],
    max_width: int,
    size: int = 28,
    fill=(255, 255, 255, 245),
    bold: bool = False,
    line_spacing: int = 6,
):
    draw = ImageDraw.Draw(image)
    font = get_font(size, bold=bold)
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if box[2] - box[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0, 220))
        box = draw.textbbox((x, y), line, font=font, stroke_width=2)
        y += (box[3] - box[1]) + line_spacing


def make_vignette(width: int, height: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / (width / 2.0)
    ny = (yy - height / 2.0) / (height / 2.0)
    radius = np.sqrt(nx * nx + ny * ny)
    return np.clip(1.0 - strength * radius**1.8, 0.0, 1.0).astype(np.float32)


def apply_grade(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(float(CONFIG["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(CONFIG["saturation"]))
    return np.array(image)


def format_srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000.0))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(captions: Sequence[Tuple[float, float, str]], path: Path) -> Path:
    lines: List[str] = []
    for index, (start, end, text) in enumerate(captions, start=1):
        lines.extend([str(index), f"{format_srt_time(start)} --> {format_srt_time(end)}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def evenly_subsample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy().reset_index(drop=True)
    indices = np.linspace(0, len(frame) - 1, maximum).astype(int)
    return frame.iloc[indices].copy().reset_index(drop=True)


def colour_ramp(value: float, alpha: int = 230) -> Tuple[int, int, int, int]:
    x = clamp(value)
    stops = [
        (0.00, COLORS["blue"]),
        (0.28, COLORS["cyan"]),
        (0.58, COLORS["violet"]),
        (0.80, COLORS["gold"]),
        (1.00, COLORS["rose"]),
    ]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= x <= b:
            u = (x - a) / max(b - a, 1e-9)
            rgb = tuple(int(round(lerp(ca[i], cb[i], u))) for i in range(3))
            return rgb + (alpha,)
    return stops[-1][1] + (alpha,)


VIGNETTE = make_vignette(OUT_W, OUT_H, float(CONFIG["vignette"]))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def build_sdss_queries() -> List[str]:
    ra_edges = np.linspace(float(CONFIG["ra_min"]), float(CONFIG["ra_max"]), int(CONFIG["ra_bins"]) + 1)
    rows_per_bin = int(math.ceil(int(CONFIG["target_rows"]) / int(CONFIG["ra_bins"])))
    queries: List[str] = []
    for lo, hi in zip(ra_edges[:-1], ra_edges[1:]):
        queries.append(
            f"""
SELECT TOP {rows_per_bin}
    specObjID,
    bestObjID,
    ra,
    dec,
    z,
    zErr,
    plate,
    mjd,
    fiberID
FROM SpecObj
WHERE class = 'GALAXY'
  AND zWarning = 0
  AND z > {float(CONFIG['z_min']):.6f}
  AND z < {float(CONFIG['z_max']):.6f}
  AND ra >= {lo:.6f}
  AND ra < {hi:.6f}
  AND dec >= {float(CONFIG['dec_min']):.6f}
  AND dec <= {float(CONFIG['dec_max']):.6f}
ORDER BY specObjID
""".strip()
        )
    return queries


def parse_skyserver_csv(payload: bytes) -> pd.DataFrame:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        raise RuntimeError("SkyServer returned an empty response")
    lowered = text.lower()
    if "<html" in lowered or "error" in lowered[:500]:
        raise RuntimeError(f"SkyServer returned an error page: {text[:240]}")
    cleaned_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    if len(cleaned_lines) < 2:
        raise RuntimeError(f"SkyServer CSV response was not a table: {text[:240]}")
    return pd.read_csv(io.StringIO("\n".join(cleaned_lines)))


def fetch_one_query(query: str) -> pd.DataFrame:
    params = urllib.parse.urlencode({"cmd": query, "format": "csv", "limit": "0"})
    url = f"{SKYSERVER_SQL_URL}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 galaxy-map-short/1.0"})
    with urllib.request.urlopen(request, timeout=float(CONFIG["data_timeout_s"])) as response:
        return parse_skyserver_csv(response.read())


def normalise_catalogue(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    rename = {column: column.strip() for column in out.columns}
    out = out.rename(columns=rename)
    lower_lookup = {column.lower(): column for column in out.columns}

    required = {}
    for canonical in ("ra", "dec", "z"):
        if canonical not in lower_lookup:
            raise RuntimeError(f"Catalogue is missing required column: {canonical}")
        required[canonical] = lower_lookup[canonical]

    optional_names = {
        "specobjid": "specObjID",
        "bestobjid": "bestObjID",
        "zerr": "zErr",
        "plate": "plate",
        "mjd": "mjd",
        "fiberid": "fiberID",
    }
    result = pd.DataFrame({
        "ra": pd.to_numeric(out[required["ra"]], errors="coerce"),
        "dec": pd.to_numeric(out[required["dec"]], errors="coerce"),
        "z": pd.to_numeric(out[required["z"]], errors="coerce"),
    })
    for lower_name, canonical in optional_names.items():
        column = lower_lookup.get(lower_name)
        if column is not None:
            result[canonical] = out[column]

    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["ra", "dec", "z"])
    result = result[
        result["ra"].between(0.0, 360.0)
        & result["dec"].between(-90.0, 90.0)
        & result["z"].between(float(CONFIG["z_min"]), float(CONFIG["z_max"]), inclusive="neither")
    ]
    if "bestObjID" in result.columns:
        best = result["bestObjID"].astype(str).str.strip()
        invalid_best = best.isin({"", "0", "0.0", "nan", "None", "null"})
        if "specObjID" in result.columns:
            fallback_key = "spec:" + result["specObjID"].astype(str)
        else:
            fallback_key = (
                "coord:"
                + result["ra"].round(7).astype(str)
                + ":"
                + result["dec"].round(7).astype(str)
                + ":"
                + result["z"].round(7).astype(str)
            )
        result["_galaxy_key"] = np.where(invalid_best, fallback_key, "photo:" + best)
        result = result.drop_duplicates("_galaxy_key").drop(columns="_galaxy_key")
    elif "specObjID" in result.columns:
        result = result.drop_duplicates("specObjID")
    else:
        result = result.drop_duplicates(["ra", "dec", "z"])
    result = result.reset_index(drop=True)
    result["data_source"] = source
    return result


def load_local_catalogue(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return normalise_catalogue(frame, "local_sdss_compatible_csv")


def fetch_sdss_catalogue() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    notes: List[str] = []
    queries = build_sdss_queries()
    pieces: List[pd.DataFrame] = []
    for index, query in enumerate(queries, start=1):
        try:
            piece = fetch_one_query(query)
            if len(piece):
                pieces.append(piece)
        except Exception as exc:
            notes.append(f"RA-bin query {index} failed: {exc}")

    if not pieces:
        raise RuntimeError("No SkyServer RA-bin query returned data")
    combined = pd.concat(pieces, ignore_index=True)
    frame = normalise_catalogue(combined, "sdss_dr16_skyserver_spectroscopy")
    if len(frame) < 500:
        raise RuntimeError(f"Only {len(frame)} valid galaxies returned")
    return frame, "sdss_dr16_skyserver_spectroscopy", notes, queries


# -----------------------------------------------------------------------------
# Deterministic offline fixture
# -----------------------------------------------------------------------------

def fallback_galaxy_catalogue() -> Tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(16012000)
    target = int(CONFIG["target_rows"])

    rows: List[Tuple[float, float, float]] = []

    # Curved filaments in sky coordinates and redshift.
    filament_count = 13
    for filament in range(filament_count):
        n = max(70, target // (filament_count * 2))
        u = rng.uniform(0.0, 1.0, n)
        ra0 = rng.uniform(float(CONFIG["ra_min"]) + 5.0, float(CONFIG["ra_max"]) - 35.0)
        width = rng.uniform(24.0, 52.0)
        dec0 = rng.uniform(2.0, 58.0)
        amp = rng.uniform(4.0, 15.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        z0 = rng.uniform(0.025, 0.18)
        dz = rng.uniform(0.025, 0.08)
        ra = ra0 + width * u + rng.normal(0.0, 1.0, n)
        dec = dec0 + amp * np.sin(2.0 * math.pi * u + phase) + rng.normal(0.0, 1.5, n)
        z = z0 + dz * (u - 0.5) + 0.008 * np.sin(4.0 * math.pi * u + phase) + rng.normal(0.0, 0.003, n)
        rows.extend(zip(ra, dec, z))

    # Cluster knots embedded in and around the filaments.
    cluster_count = 28
    for _ in range(cluster_count):
        n = max(30, target // 130)
        ra_c = rng.uniform(float(CONFIG["ra_min"]) + 4.0, float(CONFIG["ra_max"]) - 4.0)
        dec_c = rng.uniform(float(CONFIG["dec_min"]) + 5.0, float(CONFIG["dec_max"]) - 5.0)
        z_c = rng.uniform(0.02, 0.22)
        rows.extend(zip(
            rng.normal(ra_c, rng.uniform(0.4, 1.8), n),
            rng.normal(dec_c, rng.uniform(0.4, 1.8), n),
            rng.normal(z_c, rng.uniform(0.0015, 0.005), n),
        ))

    # Sparse field galaxies make the voids non-empty without erasing them.
    field_n = max(target // 5, 500)
    rows.extend(zip(
        rng.uniform(float(CONFIG["ra_min"]), float(CONFIG["ra_max"]), field_n),
        rng.uniform(float(CONFIG["dec_min"]), float(CONFIG["dec_max"]), field_n),
        rng.uniform(float(CONFIG["z_min"]), float(CONFIG["z_max"]), field_n),
    ))

    frame = pd.DataFrame(rows, columns=["ra", "dec", "z"])
    frame = frame[
        frame["ra"].between(float(CONFIG["ra_min"]), float(CONFIG["ra_max"]))
        & frame["dec"].between(float(CONFIG["dec_min"]), float(CONFIG["dec_max"]))
        & frame["z"].between(float(CONFIG["z_min"]), float(CONFIG["z_max"]))
    ]

    # Survey-shaped gaps and edges.
    gap = ((frame["ra"] > 180) & (frame["ra"] < 192) & (frame["dec"] > 34) & (frame["dec"] < 58))
    gap |= ((frame["ra"] > 232) & (frame["ra"] < 242) & (frame["dec"] < 18))
    frame = frame[~gap]

    if len(frame) > target:
        frame = frame.sample(target, random_state=1616)
    elif len(frame) < target:
        extra = frame.sample(target - len(frame), replace=True, random_state=1717).copy()
        extra["ra"] += rng.normal(0.0, 0.12, len(extra))
        extra["dec"] += rng.normal(0.0, 0.12, len(extra))
        extra["z"] += rng.normal(0.0, 0.0005, len(extra))
        frame = pd.concat([frame, extra], ignore_index=True)

    frame = frame.reset_index(drop=True)
    frame["specObjID"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    frame["zErr"] = rng.uniform(0.00002, 0.0003, len(frame))
    frame["data_source"] = "offline_cosmic_web_fixture"
    return frame, "offline_cosmic_web_fixture"


def load_all_data() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    notes: List[str] = []
    queries: List[str] = []

    if LOCAL_DATA_PATH:
        try:
            frame = load_local_catalogue(Path(LOCAL_DATA_PATH).expanduser())
            notes.append(f"Loaded GALAXY_MAP_DATA_PATH={LOCAL_DATA_PATH}")
            return frame, "local_sdss_compatible_csv", notes, queries
        except Exception as exc:
            notes.append(f"Local catalogue failed: {exc}")

    if OFFLINE_MODE:
        notes.append("Offline mode requested with GALAXY_MAP_SHORT_OFFLINE=1")
        frame, source = fallback_galaxy_catalogue()
        return frame, source, notes, queries

    try:
        frame, source, live_notes, queries = fetch_sdss_catalogue()
        notes.extend(live_notes)
        return frame, source, notes, queries
    except Exception as exc:
        notes.append(f"SDSS/SkyServer fallback: {exc}")
        frame, source = fallback_galaxy_catalogue()
        return frame, source, notes, queries


# -----------------------------------------------------------------------------
# Cosmology and map preparation
# -----------------------------------------------------------------------------

def comoving_distance_mpc(redshift: np.ndarray, h0: float, omega_m: float) -> np.ndarray:
    z = np.asarray(redshift, dtype=float)
    max_z = max(float(np.nanmax(z)), 0.001)
    grid = np.linspace(0.0, max_z * 1.002, 6000)
    omega_l = 1.0 - omega_m
    inv_e = 1.0 / np.sqrt(omega_m * (1.0 + grid) ** 3 + omega_l)
    steps = np.diff(grid)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * steps)])
    return (C_KMS / h0) * np.interp(z, grid, integral)


def choose_declination_slice(frame: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    dec = frame["dec"].to_numpy(float)
    candidates = np.linspace(max(float(CONFIG["dec_min"]), np.nanpercentile(dec, 5)), min(float(CONFIG["dec_max"]), np.nanpercentile(dec, 95)), 80)
    widths = [3.0, 5.0, 8.0, 12.0]
    best = frame.iloc[:0].copy()
    best_center = float(np.nanmedian(dec))
    best_width = widths[-1]
    for width in widths:
        for center in candidates:
            part = frame[np.abs(frame["dec"] - center) <= width / 2.0]
            if len(part) > len(best):
                best = part.copy()
                best_center = float(center)
                best_width = float(width)
        if len(best) >= max(500, len(frame) // 12):
            break
    return best.reset_index(drop=True), best_center, best_width


def prepare_catalogue(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    out = frame.copy().sort_values(["z", "ra", "dec"]).reset_index(drop=True)
    out["distance_mpc"] = comoving_distance_mpc(
        out["z"].to_numpy(float),
        float(CONFIG["reference_h0"]),
        float(CONFIG["reference_omega_m"]),
    )
    ra_rad = np.deg2rad(out["ra"].to_numpy(float))
    dec_rad = np.deg2rad(out["dec"].to_numpy(float))
    radius = out["distance_mpc"].to_numpy(float)
    out["x_mpc"] = radius * np.cos(dec_rad) * np.cos(ra_rad)
    out["y_mpc"] = radius * np.cos(dec_rad) * np.sin(ra_rad)
    out["z_mpc"] = radius * np.sin(dec_rad)

    z_lo, z_hi = np.nanpercentile(out["z"], [1.0, 99.0])
    out["z_colour"] = np.clip((out["z"] - z_lo) / max(z_hi - z_lo, 1e-9), 0.0, 1.0)

    wedge, dec_center, dec_width = choose_declination_slice(out)
    angle = np.deg2rad(wedge["ra"].to_numpy(float) - 190.0)
    distance = wedge["distance_mpc"].to_numpy(float)
    wedge["wedge_x"] = distance * np.sin(angle)
    wedge["wedge_y"] = distance * np.cos(angle)

    summary = {
        "source": str(out["data_source"].iloc[0]),
        "rows": int(len(out)),
        "redshift_min": float(out["z"].min()),
        "redshift_median": float(out["z"].median()),
        "redshift_max": float(out["z"].max()),
        "distance_min_mpc": float(out["distance_mpc"].min()),
        "distance_median_mpc": float(out["distance_mpc"].median()),
        "distance_max_mpc": float(out["distance_mpc"].max()),
        "ra_range_deg": [float(out["ra"].min()), float(out["ra"].max())],
        "dec_range_deg": [float(out["dec"].min()), float(out["dec"].max())],
        "wedge_rows": int(len(wedge)),
        "wedge_declination_center_deg": dec_center,
        "wedge_declination_width_deg": dec_width,
        "reference_cosmology": {
            "H0_km_s_Mpc": float(CONFIG["reference_h0"]),
            "Omega_m": float(CONFIG["reference_omega_m"]),
            "flat": True,
        },
    }
    return out, wedge, summary


def save_data_products(
    frame: pd.DataFrame,
    wedge: pd.DataFrame,
    summary: Dict[str, Any],
    notes: List[str],
    queries: List[str],
) -> Tuple[Path, Path]:
    catalogue_path = DATA_ROOT / "sdss_galaxy_map_catalogue.csv"
    wedge_path = DATA_ROOT / "sdss_galaxy_wedge_slice.csv"
    summary_path = DATA_ROOT / "sdss_galaxy_map_summary.json"

    frame.to_csv(catalogue_path, index=False)
    wedge.to_csv(wedge_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "notes": notes,
                "skyserver_endpoint": SKYSERVER_SQL_URL,
                "skyserver_queries": queries,
                "fallback_warning": "offline_cosmic_web_fixture is deterministic synthetic preview data, not observational data",
                "science_warning": "survey footprint edges and gaps are selection geometry, not physical boundaries",
                "source_urls": {
                    "sdss_dr16_api": "https://cas.sdss.org/dr16/en/help/docs/api.aspx",
                    "sdss_dr16_skyserver": "https://skyserver.sdss.org/dr16/en/home.aspx",
                    "sdss_data_releases": "https://www.sdss.org/science/publications/data-release-publications/",
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return catalogue_path, summary_path


def create_scientific_plots(frame: pd.DataFrame, wedge: pd.DataFrame):
    display = evenly_subsample(frame, 16000)

    fig, ax = plt.subplots(figsize=(10, 5.3))
    scatter = ax.scatter(display["ra"], display["dec"], c=display["z"], s=2, alpha=0.65)
    ax.set_title("SDSS spectroscopic galaxy sample")
    ax.set_xlabel("Right ascension (degrees)")
    ax.set_ylabel("Declination (degrees)")
    fig.colorbar(scatter, ax=ax, label="Redshift")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "sky_footprint_redshift.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    scatter = ax.scatter(wedge["wedge_x"], wedge["wedge_y"], c=wedge["z"], s=2, alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Thin redshift wedge")
    ax.set_xlabel("Transverse reference comoving distance (Mpc)")
    ax.set_ylabel("Radial reference comoving distance (Mpc)")
    fig.colorbar(scatter, ax=ax, label="Redshift")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "redshift_wedge.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(frame["z"], bins=50)
    ax.set_title("Redshift distribution")
    ax.set_xlabel("Spectroscopic redshift")
    ax.set_ylabel("Galaxies")
    plt.tight_layout()
    plt.savefig(PREVIEW_DIR / "redshift_histogram.png", dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scene renderer
# -----------------------------------------------------------------------------

class GalaxyMapScene:
    def __init__(self, frame: pd.DataFrame, wedge: pd.DataFrame, summary: Dict[str, Any]):
        self.frame = evenly_subsample(frame, int(CONFIG["max_render_points"]))
        self.wedge = evenly_subsample(wedge, min(int(CONFIG["max_render_points"]), 5200 if not QUICK_MODE else 1700))
        self.summary = summary
        self.stars = self._make_stars(int(CONFIG["background_stars"]), seed=194)
        self.hud = self._make_hud(int(CONFIG["hud_noise"]), seed=206)

        self.ra_min = float(frame["ra"].min())
        self.ra_max = float(frame["ra"].max())
        self.dec_min = float(frame["dec"].min())
        self.dec_max = float(frame["dec"].max())
        self.z_min = float(frame["z"].min())
        self.z_max = float(frame["z"].max())

        # Pre-sort for coherent reveals.
        self.sky_order = self.frame.sort_values(["ra", "dec"]).reset_index(drop=True)
        self.depth_order = self.frame.sort_values("z").reset_index(drop=True)
        self.wedge_order = self.wedge.sort_values(["wedge_y", "wedge_x"]).reset_index(drop=True)

        xyz = self.frame[["x_mpc", "y_mpc", "z_mpc"]].to_numpy(float)
        xyz -= np.nanmean(xyz, axis=0, keepdims=True)
        scale = max(float(np.nanpercentile(np.linalg.norm(xyz, axis=1), 98.0)), 1.0)
        self.xyz = xyz / scale

    @staticmethod
    def _make_stars(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "r": float(rng.uniform(0.4, 1.8)),
                "a": float(rng.uniform(16, 90)),
                "phase": float(rng.uniform(0, 2.0 * math.pi)),
            }
            for _ in range(count)
        ]

    @staticmethod
    def _make_hud(count: int, seed: int) -> List[Dict[str, float]]:
        rng = np.random.default_rng(seed)
        return [
            {
                "x": float(rng.uniform(0, OUT_W)),
                "y": float(rng.uniform(0, OUT_H)),
                "length": float(rng.uniform(10, 90)),
                "a": float(rng.uniform(8, 38)),
                "phase": float(rng.uniform(0, 2.0 * math.pi)),
            }
            for _ in range(count)
        ]

    def background(self, t: float) -> Image.Image:
        image = Image.new("RGBA", OUT_SIZE, (2, 6, 15, 255))
        draw = ImageDraw.Draw(image)
        for star in self.stars:
            alpha = int(star["a"] * (0.72 + 0.28 * math.sin(t * 1.5 + star["phase"])))
            radius = star["r"]
            draw.ellipse(
                (star["x"] - radius, star["y"] - radius, star["x"] + radius, star["y"] + radius),
                fill=(220, 235, 255, alpha),
            )

        haze = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        clouds = [
            (OUT_W * 0.22, OUT_H * 0.30, (25, 50, 145)),
            (OUT_W * 0.78, OUT_H * 0.38, (55, 22, 110)),
            (OUT_W * 0.52, OUT_H * 0.78, (8, 76, 120)),
        ]
        for cx, cy, colour in clouds:
            for radius, alpha in [(420 * OUT_W / 1080.0, 14), (280 * OUT_W / 1080.0, 22), (170 * OUT_W / 1080.0, 29)]:
                hd.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour + (alpha,))
        haze = haze.filter(ImageFilter.GaussianBlur(62 if not QUICK_MODE else 31))
        image.alpha_composite(haze)
        return image

    @staticmethod
    def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 168):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle(
            box,
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 7, 17, alpha),
            outline=(100, 200, 235, 62),
            width=1,
        )
        image.alpha_composite(overlay)

    def map_sky(self, ra: float, dec: float, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x0, y0, x1, y1 = box
        x = x1 - (ra - self.ra_min) / max(self.ra_max - self.ra_min, 1e-9) * (x1 - x0)
        y = y1 - (dec - self.dec_min) / max(self.dec_max - self.dec_min, 1e-9) * (y1 - y0)
        return x, y

    def draw_map_grid(self, image: Image.Image, box: Tuple[int, int, int, int]):
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = box
        for i in range(1, 5):
            x = lerp(x0, x1, i / 5.0)
            draw.line((x, y0, x, y1), fill=(160, 215, 235, 30), width=1)
        for i in range(1, 4):
            y = lerp(y0, y1, i / 4.0)
            draw.line((x0, y, x1, y), fill=(160, 215, 235, 30), width=1)

    def draw_intro(self, image: Image.Image, t: float):
        shot = SHOT_PLAN[0]
        local = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"], 1e-9))
        cx, cy = OUT_W * 0.5, OUT_H * 0.39
        draw = ImageDraw.Draw(image)

        count = max(20, int(len(self.sky_order) * min(local * 1.35, 1.0)))
        part = self.sky_order.iloc[:count]
        box = (int(OUT_W * 0.14), int(OUT_H * 0.22), int(OUT_W * 0.86), int(OUT_H * 0.57))
        for index, row in part.iterrows():
            tx, ty = self.map_sky(float(row["ra"]), float(row["dec"]), box)
            seed_phase = (index * 0.61803398875) % 1.0
            sx = OUT_W * (0.06 + 0.88 * seed_phase)
            sy = OUT_H * (0.12 + 0.55 * ((index * 0.41421356237) % 1.0))
            u = smoothstep(clamp(local * 1.35 - (index / max(len(self.sky_order), 1)) * 0.72))
            x = lerp(sx, tx, u)
            y = lerp(sy, ty, u)
            colour = colour_ramp(float(row["z_colour"]), 185)
            rr = 1.7 if not QUICK_MODE else 1.0
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=colour)

        draw_text(
            image,
            f"{len(self.frame):,} SPECTRA → ONE STRUCTURE MAP",
            (OUT_W // 2, int(OUT_H * 0.65)),
            size=27 if not QUICK_MODE else 13,
            fill=COLORS["gold"] + (240,),
            bold=True,
            anchor="ma",
            stroke=1,
        )
        draw_text(
            image,
            "each point is one measured galaxy",
            (OUT_W // 2, int(OUT_H * 0.70)),
            size=20 if not QUICK_MODE else 10,
            fill=COLORS["white"] + (220,),
            anchor="ma",
            stroke=1,
        )

    def draw_sky_map(self, image: Image.Image, t: float, coloured: bool = False):
        box_outer = (int(OUT_W * 0.06), int(OUT_H * 0.22), int(OUT_W * 0.94), int(OUT_H * 0.77))
        self.panel(image, box_outer)
        box = (box_outer[0] + 24, box_outer[1] + 78, box_outer[2] - 20, box_outer[3] - 42)
        self.draw_map_grid(image, box)
        draw = ImageDraw.Draw(image)

        shot = next(item for item in SHOT_PLAN if item["name"] == ("redshift" if coloured else "sky_map"))
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.7, 1e-9))
        ordered = self.depth_order if coloured else self.sky_order
        count = max(20, int(len(ordered) * reveal))
        part = ordered.iloc[:count]

        for _, row in part.iterrows():
            x, y = self.map_sky(float(row["ra"]), float(row["dec"]), box)
            colour = colour_ramp(float(row["z_colour"]), 205) if coloured else COLORS["ice"] + (180,)
            rr = 1.9 if not QUICK_MODE else 1.0
            draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=colour)

        title = "THE SDSS SURVEY FOOTPRINT" if not coloured else "THE SAME MAP, COLOURED BY REDSHIFT"
        subtitle = "RA + DEC // direction on the sky" if not coloured else "blue = nearer // warm = farther"
        draw_text(image, title, (box_outer[0] + 24, box_outer[1] + 18), size=22 if not QUICK_MODE else 11,
                  fill=(COLORS["cyan"] if not coloured else COLORS["violet"]) + (240,), bold=True, stroke=1)
        draw_text(image, subtitle, (box_outer[0] + 24, box_outer[1] + 50), size=16 if not QUICK_MODE else 8,
                  fill=COLORS["white"] + (210,), stroke=1)
        draw_text(image, "RIGHT ASCENSION ←", (box[2], box_outer[3] - 16), size=14 if not QUICK_MODE else 7,
                  fill=COLORS["muted"] + (190,), anchor="ra", stroke=1)
        draw_text(image, "DECLINATION", (box[0] - 6, box[1] - 8), size=13 if not QUICK_MODE else 6,
                  fill=COLORS["muted"] + (180,), stroke=1)

        if coloured:
            legend_x0 = box[0]
            legend_x1 = box[0] + int((box[2] - box[0]) * 0.50)
            legend_y = box_outer[3] - (24 if not QUICK_MODE else 12)
            steps = 90
            for i in range(steps):
                xa = lerp(legend_x0, legend_x1, i / steps)
                xb = lerp(legend_x0, legend_x1, (i + 1) / steps)
                draw.rectangle((xa, legend_y - 7, xb + 1, legend_y + 2), fill=colour_ramp(i / (steps - 1), 230))
            draw_text(image, f"z {self.z_min:.2f}", (legend_x0, legend_y + 8), size=12 if not QUICK_MODE else 6,
                      fill=COLORS["muted"] + (205,), stroke=1)
            draw_text(image, f"z {self.z_max:.2f}", (legend_x1, legend_y + 8), size=12 if not QUICK_MODE else 6,
                      fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)

    def draw_wedge(self, image: Image.Image, t: float):
        box_outer = (int(OUT_W * 0.05), int(OUT_H * 0.20), int(OUT_W * 0.95), int(OUT_H * 0.80))
        self.panel(image, box_outer)
        box = (box_outer[0] + 22, box_outer[1] + 82, box_outer[2] - 22, box_outer[3] - 38)
        draw = ImageDraw.Draw(image)

        x_values = self.wedge_order["wedge_x"].to_numpy(float)
        y_values = self.wedge_order["wedge_y"].to_numpy(float)
        x_limit = max(float(np.nanpercentile(np.abs(x_values), 99.0)), 1.0)
        y_min = max(0.0, float(np.nanpercentile(y_values, 1.0)))
        y_max = float(np.nanpercentile(y_values, 99.0))

        # Fan guide lines.
        origin_x = (box[0] + box[2]) / 2.0
        origin_y = box[3]
        for frac in (-0.8, -0.4, 0.0, 0.4, 0.8):
            draw.line((origin_x, origin_y, lerp(box[0], box[2], (frac + 1.0) / 2.0), box[1]), fill=(160, 215, 235, 28), width=1)
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = lerp(box[3], box[1], frac)
            half = (box[2] - box[0]) * 0.5 * frac
            draw.arc((origin_x - half, y - 10, origin_x + half, y + 10), 180, 360, fill=(160, 215, 235, 25), width=1)

        shot = next(item for item in SHOT_PLAN if item["name"] == "wedge")
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.8, 1e-9))
        count = max(20, int(len(self.wedge_order) * reveal))
        part = self.wedge_order.iloc[:count]
        for _, row in part.iterrows():
            x = lerp(box[0], box[2], (float(row["wedge_x"]) / x_limit + 1.0) / 2.0)
            y = lerp(box[3], box[1], (float(row["wedge_y"]) - y_min) / max(y_max - y_min, 1e-9))
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                rr = 2.0 if not QUICK_MODE else 1.0
                draw.ellipse((x - rr, y - rr, x + rr, y + rr), fill=colour_ramp(float(row["z_colour"]), 205))

        draw_text(image, "A THIN REDSHIFT WEDGE", (box_outer[0] + 24, box_outer[1] + 18), size=23 if not QUICK_MODE else 11,
                  fill=COLORS["gold"] + (240,), bold=True, stroke=1)
        draw_text(
            image,
            f"declination {self.summary['wedge_declination_center_deg']:+.1f}° ± {self.summary['wedge_declination_width_deg'] / 2.0:.1f}°",
            (box_outer[0] + 24, box_outer[1] + 51),
            size=16 if not QUICK_MODE else 8,
            fill=COLORS["white"] + (210,),
            stroke=1,
        )
        draw_text(image, "observer", (int(origin_x), box[3] + 12), size=13 if not QUICK_MODE else 6,
                  fill=COLORS["muted"] + (190,), anchor="ma", stroke=1)
        draw_text(image, "walls + filaments + voids", (box[2], box_outer[3] - 17), size=15 if not QUICK_MODE else 7,
                  fill=COLORS["cyan"] + (220,), bold=True, anchor="ra", stroke=1)

    def project_xyz(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        angle_y = 0.4 + t * 0.16
        angle_x = -0.38 + 0.07 * math.sin(t * 0.45)
        cy, sy = math.cos(angle_y), math.sin(angle_y)
        cx, sx = math.cos(angle_x), math.sin(angle_x)
        rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rotated = self.xyz @ rotation_y.T @ rotation_x.T
        perspective = 1.0 / np.clip(2.3 - rotated[:, 2] * 0.55, 1.2, 3.0)
        return rotated[:, 0] * perspective, rotated[:, 1] * perspective, rotated[:, 2]

    def draw_point_cloud(self, image: Image.Image, t: float, finale: bool = False):
        x, y, depth = self.project_xyz(t)
        order = np.argsort(depth)
        draw = ImageDraw.Draw(image)
        shot_name = "finale" if finale else "point_cloud"
        shot = next(item for item in SHOT_PLAN if item["name"] == shot_name)
        reveal = smoothstep((t - shot["start"]) / max(shot["end"] - shot["start"] - 0.7, 1e-9))
        visible = max(20, int(len(order) * reveal))
        order = order[:visible]

        cx = OUT_W * 0.5
        cy = OUT_H * (0.43 if not finale else 0.40)
        scale = OUT_W * (0.76 if not finale else 0.82)

        glow = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for index in order[:: max(1, len(order) // 1500)]:
            px = cx + x[index] * scale
            py = cy - y[index] * scale
            colour = colour_ramp(float(self.frame.iloc[index]["z_colour"]), 30)
            rr = 5.0 if not QUICK_MODE else 2.5
            gd.ellipse((px - rr, py - rr, px + rr, py + rr), fill=colour)
        glow = glow.filter(ImageFilter.GaussianBlur(5 if not QUICK_MODE else 2))
        image.alpha_composite(glow)

        draw = ImageDraw.Draw(image)
        for index in order:
            px = cx + x[index] * scale
            py = cy - y[index] * scale
            depth_scale = clamp((depth[index] + 1.0) / 2.0)
            rr = (1.2 + 1.8 * depth_scale) if not QUICK_MODE else (0.7 + 0.8 * depth_scale)
            colour = colour_ramp(float(self.frame.iloc[index]["z_colour"]), int(145 + 90 * depth_scale))
            draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=colour)

        if not finale:
            draw_text(image, "REDSHIFT → REFERENCE 3D DISTANCE", (OUT_W // 2, int(OUT_H * 0.70)), size=24 if not QUICK_MODE else 12,
                      fill=COLORS["violet"] + (240,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "a rotating point cloud of measured galaxies", (OUT_W // 2, int(OUT_H * 0.75)), size=18 if not QUICK_MODE else 9,
                      fill=COLORS["white"] + (215,), anchor="ma", stroke=1)
        else:
            self.panel(image, (int(OUT_W * 0.08), int(OUT_H * 0.62), int(OUT_W * 0.92), int(OUT_H * 0.80)), alpha=178)
            draw_text(image, "THE COSMIC WEB", (OUT_W // 2, int(OUT_H * 0.655)), size=31 if not QUICK_MODE else 15,
                      fill=COLORS["cyan"] + (245,), bold=True, anchor="ma", stroke=1)
            draw_text(image, f"{self.summary['rows']:,} galaxies // z {self.summary['redshift_min']:.3f}–{self.summary['redshift_max']:.3f}",
                      (OUT_W // 2, int(OUT_H * 0.705)), size=20 if not QUICK_MODE else 10,
                      fill=COLORS["gold"] + (235,), bold=True, anchor="ma", stroke=1)
            draw_text(image, "one survey map, part of a much larger universe", (OUT_W // 2, int(OUT_H * 0.750)),
                      size=17 if not QUICK_MODE else 8, fill=COLORS["white"] + (220,), anchor="ma", stroke=1)

    def draw_source_hud(self, image: Image.Image):
        live = self.summary["source"] in {"sdss_dr16_skyserver_spectroscopy", "local_sdss_compatible_csv"}
        label = "SOURCE // SDSS DR16 SPECTROSCOPY" if live else "PREVIEW SOURCE // SYNTHETIC FIXTURE"
        colour = COLORS["cyan"] if live else COLORS["gold"]
        draw_text(image, label, (OUT_W - (46 if not QUICK_MODE else 23), 70 if not QUICK_MODE else 35),
                  size=17 if not QUICK_MODE else 8, fill=colour + (235,), bold=True, anchor="ra", stroke=1)
        draw_text(image, f"GALAXIES // {self.summary['rows']:,}", (OUT_W - (46 if not QUICK_MODE else 23), 100 if not QUICK_MODE else 50),
                  size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (205,), anchor="ra", stroke=1)
        draw_text(image, f"MEDIAN z // {self.summary['redshift_median']:.3f}", (OUT_W - (46 if not QUICK_MODE else 23), 127 if not QUICK_MODE else 63),
                  size=15 if not QUICK_MODE else 7, fill=COLORS["muted"] + (195,), anchor="ra", stroke=1)

    def draw_titles(self, image: Image.Image, t: float, shot_name: str):
        intro_end = 6.7 if not QUICK_MODE else 1.4
        alpha = int(255 * smoothstep((t - 0.2) / 0.8) * (1.0 - smoothstep((t - intro_end) / 0.65)))
        if alpha > 4:
            draw_text(image, "THOUSANDS OF GALAXIES,", (54 if not QUICK_MODE else 27, 88 if not QUICK_MODE else 43),
                      size=40 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, "ONE MAP", (54 if not QUICK_MODE else 27, 136 if not QUICK_MODE else 67),
                      size=40 if not QUICK_MODE else 19, fill=COLORS["white"] + (alpha,), bold=True)
            draw_text(image, CONFIG["subtitle"], (56 if not QUICK_MODE else 28, 188 if not QUICK_MODE else 94),
                      size=21 if not QUICK_MODE else 10, fill=COLORS["cyan"] + (min(alpha, 230),), bold=True)

        labels = {
            "intro": "SPECTRA BECOME COORDINATES",
            "sky_map": "SKY POSITION // THE SURVEY FOOTPRINT",
            "redshift": "REDSHIFT // ADDING DEPTH",
            "wedge": "A THIN SLICE // STRUCTURE EMERGES",
            "point_cloud": "REFERENCE 3D MAP // ROTATING THE SURVEY",
            "finale": "GRAVITY'S LARGE-SCALE ARCHITECTURE",
        }
        if t > (5.1 if not QUICK_MODE else 1.2):
            draw_text(image, labels[shot_name], (54 if not QUICK_MODE else 27, 60 if not QUICK_MODE else 30),
                      size=18 if not QUICK_MODE else 9, fill=COLORS["muted"] + (205,), bold=True, stroke=1)

    def draw_caption(self, image: Image.Image, t: float):
        text = caption_at(t)
        if not text:
            return
        y0 = OUT_H - (244 if not QUICK_MODE else 124)
        panel = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (44 if not QUICK_MODE else 22, y0, OUT_W - (44 if not QUICK_MODE else 22), y0 + (124 if not QUICK_MODE else 66)),
            radius=24 if not QUICK_MODE else 12,
            fill=(2, 6, 15, 174),
            outline=(80, 190, 228, 65),
            width=1,
        )
        image.alpha_composite(panel)
        draw_wrapped_text(
            image,
            text,
            (68 if not QUICK_MODE else 34, y0 + (28 if not QUICK_MODE else 14)),
            OUT_W - (136 if not QUICK_MODE else 68),
            size=29 if not QUICK_MODE else 14,
            fill=COLORS["white"] + (245,),
        )

    def draw_hud_noise(self, image: Image.Image, t: float):
        overlay = Image.new("RGBA", OUT_SIZE, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for item in self.hud:
            pulse = 0.5 + 0.5 * math.sin(t * 1.9 + item["phase"])
            if pulse < 0.74:
                continue
            y = (item["y"] + t * 9.0) % OUT_H
            draw.line((item["x"], y, item["x"] + item["length"], y), fill=COLORS["cyan"] + (int(item["a"] * pulse),), width=1)
        offset = int((t * 39) % 7)
        for y in range(offset, OUT_H, 7):
            draw.line((0, y, OUT_W, y), fill=(120, 200, 240, 10), width=1)
        scan_y = int((t * 164) % (OUT_H + 220)) - 110
        draw.rectangle((0, scan_y, OUT_W, scan_y + (48 if not QUICK_MODE else 24)), fill=(80, 210, 240, 8))
        image.alpha_composite(overlay)

    def render_frame(self, t: float) -> np.ndarray:
        shot = get_shot(t)
        name = shot["name"]
        image = self.background(t)

        if name == "intro":
            self.draw_intro(image, t)
        elif name == "sky_map":
            self.draw_sky_map(image, t, coloured=False)
        elif name == "redshift":
            self.draw_sky_map(image, t, coloured=True)
        elif name == "wedge":
            self.draw_wedge(image, t)
        elif name == "point_cloud":
            self.draw_point_cloud(image, t, finale=False)
        elif name == "finale":
            self.draw_point_cloud(image, t, finale=True)

        self.draw_source_hud(image)
        self.draw_titles(image, t, name)
        self.draw_caption(image, t)
        self.draw_hud_noise(image, t)

        array = np.array(image.convert("RGB"))
        array = apply_grade(array)
        array = np.clip(array.astype(np.float32) * VIGNETTE[..., None], 0, 255).astype(np.uint8)
        fade_in = smoothstep(t / 0.9)
        fade_out = 1.0 - smoothstep((t - (float(CONFIG["duration_s"]) - 1.1)) / 1.0)
        return np.clip(array.astype(np.float32) * fade_in * fade_out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_video(scene: GalaxyMapScene) -> Path:
    srt_path = OUTPUT_ROOT / f"{CONFIG['output_basename']}.srt"
    write_srt(CAPTIONS, srt_path)
    print("Subtitle sidecar:", srt_path.resolve())

    raw_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_raw.mp4"
    final_video = OUTPUT_ROOT / f"{CONFIG['output_basename']}_final.mp4"
    frame_count = int(round(float(CONFIG["duration_s"]) * int(CONFIG["fps"])))
    times = np.arange(frame_count) / int(CONFIG["fps"])
    print(f"Rendering {frame_count:,} frames at {OUT_W}x{OUT_H} ...")
    with iio.get_writer(
        raw_video,
        fps=int(CONFIG["fps"]),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for t in tqdm(times, desc="Rendering galaxy-map short"):
            writer.append_data(scene.render_frame(float(t)))
    shutil.copyfile(raw_video, final_video)
    print("Final video:", final_video.resolve())
    return final_video


def main():
    print("Loading galaxy spectroscopy ...")
    frame, source, notes, queries = load_all_data()

    print("Converting redshift to reference comoving distance ...")
    frame, wedge, summary = prepare_catalogue(frame)
    catalogue_path, summary_path = save_data_products(frame, wedge, summary, notes, queries)
    create_scientific_plots(frame, wedge)

    print("Catalogue source:", source)
    print("Galaxies:", f"{len(frame):,}")
    print("Redshift range:", f"{frame['z'].min():.5f} to {frame['z'].max():.5f}")
    print("Wedge galaxies:", f"{len(wedge):,}")
    for note in notes:
        print("Data note:", note)
    print("Data:", catalogue_path.resolve())
    print("Summary:", summary_path.resolve())

    scene = GalaxyMapScene(frame, wedge, summary)
    preview_times = [
        1.0,
        min(10.0, float(CONFIG["duration_s"]) * 0.20),
        min(22.0, float(CONFIG["duration_s"]) * 0.39),
        min(34.0, float(CONFIG["duration_s"]) * 0.60),
        min(45.0, float(CONFIG["duration_s"]) * 0.79),
        float(CONFIG["duration_s"]) - 1.0,
    ]
    for preview_time in tqdm(preview_times, desc="Preview frames"):
        frame_image = scene.render_frame(float(preview_time))
        Image.fromarray(frame_image).save(PREVIEW_DIR / f"preview_{int(preview_time):02d}s.png")

    render_video(scene)
    print("Output directory:", OUTPUT_ROOT.resolve())
    for path in sorted(OUTPUT_ROOT.glob("*")):
        print("-", path.name)


if __name__ == "__main__":
    main()
