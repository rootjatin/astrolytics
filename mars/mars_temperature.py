from __future__ import annotations

"""
The Real Temperature of Mars for One Year
==========================================

Cinematic vertical YouTube Shorts renderer (1080x1920, optional 4K) using
published Curiosity/REMS temperature ranges from Gale Crater.

The default dataset is a 12-bin digitization of NASA's infographic
"Seasonal Cycles in Curiosity's First Two Martian Years". NASA states that
one bar represents an average over one-twelfth of a Martian year. This script
uses the first 12 bars (Curiosity's first Martian year) and keeps the scope
explicitly location-specific: air temperature at Gale Crater, not a global
Mars temperature.

Published source:
https://science.nasa.gov/resource/seasonal-cycles-in-curiositys-first-two-martian-years/

Underlying instrument archive:
NASA PDS, MSL-M-REMS-4-ENVRDR-V1.0
https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=MSL-M-REMS-4-ENVRDR-V1.0

Modes
-----
Standard 52-second render:
    python mars_temperature_one_year_cinematic_short.py

Fast 13-second preview:
    MARS_TEMP_SHORT_QUICK=1 python mars_temperature_one_year_cinematic_short.py

True 4K vertical:
    MARS_TEMP_SHORT_4K=1 python mars_temperature_one_year_cinematic_short.py

Use a replacement CSV with columns sol_center,max_c,min_c,season:
    MARS_TEMP_DATA_PATH=/path/to/temperature_bins.csv \
      python mars_temperature_one_year_cinematic_short.py

Disable soundtrack:
    MARS_TEMP_SHORT_SOUND=0 python mars_temperature_one_year_cinematic_short.py
"""

import csv
import json
import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration and real published data
# -----------------------------------------------------------------------------

QUICK = os.environ.get("MARS_TEMP_SHORT_QUICK", "0") == "1"
FOUR_K = os.environ.get("MARS_TEMP_SHORT_4K", "0") == "1" and not QUICK
SOUND = os.environ.get("MARS_TEMP_SHORT_SOUND", "1") != "0"
LOCAL_DATA = os.environ.get("MARS_TEMP_DATA_PATH", "").strip()

W = 540 if QUICK else (2160 if FOUR_K else 1080)
H = 960 if QUICK else (3840 if FOUR_K else 1920)
FPS = 8 if QUICK else (30 if FOUR_K else 24)
DURATION = 13.0 if QUICK else 52.0
SIZE = (W, H)
SCALE = W / 1080.0

OUTPUT = Path("mars_temperature_one_year_output")
DATA_DIR = OUTPUT / "data"
PREVIEW_DIR = OUTPUT / "previews"
for directory in (OUTPUT, DATA_DIR, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BASENAME = "the_real_temperature_of_mars_for_one_year"

# First 12 published one-twelfth-year ranges, digitized from NASA/JPL's graphic.
# Values are approximately ±1 °C because they are read from a plotted graphic,
# not recomputed from the raw one-second REMS tables.
DEFAULT_BINS = [
    # sol_center, max air temperature °C, min air temperature °C, season
    (27.8, -9.3, -71.2, "Spring"),
    (83.5, -5.4, -69.5, "Spring"),
    (139.1, -3.6, -64.7, "Spring"),
    (194.8, -3.5, -62.8, "Summer"),
    (250.4, -4.7, -63.7, "Summer"),
    (306.1, -5.6, -65.5, "Summer"),
    (361.7, -5.5, -66.7, "Autumn"),
    (417.4, -8.9, -72.4, "Autumn"),
    (473.0, -19.1, -75.3, "Autumn"),
    (528.7, -27.9, -80.5, "Winter"),
    (584.3, -27.1, -80.8, "Winter"),
    (640.0, -22.3, -78.4, "Winter"),
]

SOURCE_URLS = {
    "nasa_infographic": "https://science.nasa.gov/resource/seasonal-cycles-in-curiositys-first-two-martian-years/",
    "nasa_image": "https://assets.science.nasa.gov/dynamicimage/assets/science/psd/mars/downloadable_items/3/9/39185_mars-curiosity-atmosphere-pia20600.jpg",
    "pds_rems_dataset": "https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=MSL-M-REMS-4-ENVRDR-V1.0",
    "mars_facts": "https://science.nasa.gov/mars/facts/",
}

COLORS = {
    "white": (247, 245, 239),
    "muted": (193, 191, 186),
    "red": (230, 86, 46),
    "orange": (255, 145, 55),
    "gold": (255, 202, 100),
    "cyan": (105, 222, 235),
    "ice": (170, 229, 244),
    "blue": (64, 126, 190),
    "deep": (4, 7, 14),
    "dust": (177, 91, 51),
    "sand": (138, 72, 43),
    "night": (10, 14, 26),
}

FULL_CAPTIONS = [
    (0.4, 5.4, "A year on Mars lasts 687 Earth days. But Mars does not have one single temperature."),
    (5.5, 13.2, "At Gale Crater, Curiosity measured the air through one complete Martian year — from spring, through winter, and back toward spring."),
    (13.3, 22.3, "The warmest one-twelfth-year average afternoon reached about minus three Celsius. The coldest average night fell below minus eighty."),
    (22.4, 31.8, "On the same part of Mars, afternoon and night were often separated by more than sixty Celsius degrees."),
    (31.9, 42.3, "Spring and summer were least cold. Autumn and winter plunged lower as Mars moved through its long, uneven orbit."),
    (42.4, 51.4, "This is a real rover record from Gale Crater — not a global average, and not the temperature everywhere on Mars."),
]

SHOT_PLAN_FULL = [
    ("intro", 0.0, 5.5),
    ("year", 5.5, 13.3),
    ("extremes", 13.3, 22.4),
    ("annual", 22.4, 31.9),
    ("seasons", 31.9, 42.4),
    ("finale", 42.4, 52.0),
]

if QUICK:
    factor = DURATION / 52.0
    CAPTIONS = [(a * factor, b * factor, text) for a, b, text in FULL_CAPTIONS]
    SHOT_PLAN = [(name, a * factor, b * factor) for name, a, b in SHOT_PLAN_FULL]
else:
    CAPTIONS = FULL_CAPTIONS
    SHOT_PLAN = SHOT_PLAN_FULL


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TempBin:
    sol_center: float
    max_c: float
    min_c: float
    season: str


def load_bins() -> List[TempBin]:
    if not LOCAL_DATA:
        return [TempBin(*row) for row in DEFAULT_BINS]
    rows: List[TempBin] = []
    with Path(LOCAL_DATA).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sol_center", "max_c", "min_c", "season"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Replacement CSV is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(TempBin(float(row["sol_center"]), float(row["max_c"]), float(row["min_c"]), row["season"]))
    if len(rows) < 4:
        raise ValueError("Replacement CSV must contain at least four temperature bins")
    return rows


BINS = load_bins()
SOL = np.array([b.sol_center for b in BINS], dtype=np.float64)
MAX_C = np.array([b.max_c for b in BINS], dtype=np.float64)
MIN_C = np.array([b.min_c for b in BINS], dtype=np.float64)
MEAN_C = (MAX_C + MIN_C) / 2.0
RANGE_C = MAX_C - MIN_C

STATS = {
    "warmest_bin_max_c": float(MAX_C.max()),
    "coldest_bin_min_c": float(MIN_C.min()),
    "largest_bin_range_c": float(RANGE_C.max()),
    "mean_of_bin_midpoints_c": float(MEAN_C.mean()),
    "bin_count": len(BINS),
    "martian_year_sols_approx": 668.6,
    "martian_year_earth_days": 687,
    "location": "Gale Crater",
    "instrument": "Curiosity Rover Environmental Monitoring Station (REMS)",
    "data_note": "Approximate digitization of the first 12 one-twelfth-year temperature-range bars in NASA/JPL-Caltech/CAB(CSIC-INTA) infographic PIA20600.",
}


def write_data_products() -> Tuple[Path, Path]:
    csv_path = DATA_DIR / "curiosity_first_martian_year_temperature_bins.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin", "sol_center", "max_air_temperature_c", "min_air_temperature_c", "midpoint_c", "range_c", "season"])
        for index, b in enumerate(BINS, 1):
            writer.writerow([index, f"{b.sol_center:.1f}", f"{b.max_c:.1f}", f"{b.min_c:.1f}", f"{(b.max_c+b.min_c)/2:.1f}", f"{b.max_c-b.min_c:.1f}", b.season])
    json_path = DATA_DIR / "method_and_sources.json"
    json_path.write_text(json.dumps({"stats": STATS, "sources": SOURCE_URLS}, indent=2), encoding="utf-8")
    return csv_path, json_path


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(x: float) -> float:
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def shot_at(t: float) -> Tuple[str, float, float]:
    for shot in SHOT_PLAN:
        if shot[1] <= t < shot[2]:
            return shot
    return SHOT_PLAN[-1]


def caption_at(t: float) -> str | None:
    for a, b, text in CAPTIONS:
        if a <= t < b:
            return text
    return None


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, max(6, int(size * SCALE)))
        except Exception:
            pass
    return ImageFont.load_default()


def text(image: Image.Image, value: str, xy: Tuple[float, float], size: int, fill=(255, 255, 255, 255), bold=False, anchor="la", stroke=2) -> None:
    ImageDraw.Draw(image).text(
        (int(xy[0]), int(xy[1])), value, font=font(size, bold), fill=fill,
        anchor=anchor, stroke_width=max(1, int(stroke * SCALE)), stroke_fill=(0, 0, 0, 210),
    )


def wrapped(image: Image.Image, value: str, box: Tuple[int, int, int, int], size: int, fill=(255,255,255,245), bold=False, spacing=6, align="center") -> None:
    x0, y0, x1, _ = box
    draw = ImageDraw.Draw(image)
    fnt = font(size, bold)
    words = value.split()
    lines: List[str] = []
    current = ""
    max_width = x1 - x0
    for word in words:
        candidate = word if not current else current + " " + word
        if draw.textbbox((0,0), candidate, font=fnt, stroke_width=max(1,int(1*SCALE)))[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    y = y0
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=fnt, stroke_width=max(1,int(1*SCALE)))
        h = bbox[3] - bbox[1]
        x = (x0+x1)//2 if align == "center" else x0
        anchor = "ma" if align == "center" else "la"
        draw.text((x,y), line, font=fnt, fill=fill, anchor=anchor, stroke_width=max(1,int(2*SCALE)), stroke_fill=(0,0,0,210))
        y += h + int(spacing*SCALE)


def panel(image: Image.Image, box: Tuple[int, int, int, int], alpha: int = 165, radius: int = 26) -> None:
    layer = Image.new("RGBA", SIZE, (0,0,0,0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=max(6,int(radius*SCALE)), fill=(4,7,15,alpha), outline=(255,184,108,55), width=max(1,int(SCALE)))
    image.alpha_composite(layer)


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> Path:
    lines: List[str] = []
    for i, (a,b,value) in enumerate(CAPTIONS, 1):
        lines += [str(i), f"{srt_time(a)} --> {srt_time(b)}", value, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Scene assets and renderer
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Star:
    x: float
    y: float
    r: float
    a: int
    phase: float


class Renderer:
    def __init__(self) -> None:
        rng = np.random.default_rng(20260806)
        self.stars = [Star(float(rng.uniform(0,W)), float(rng.uniform(0,H)), float(rng.uniform(.35,2.0)*SCALE), int(rng.uniform(35,170)), float(rng.uniform(0,2*math.pi))) for _ in range(240 if QUICK else 620)]
        self.craters = [(float(rng.uniform(-.75,.75)), float(rng.uniform(-.72,.72)), float(rng.uniform(.025,.13)), float(rng.uniform(.3,.9))) for _ in range(44)]
        self.dust = [(float(rng.uniform(0,W)), float(rng.uniform(0,H)), float(rng.uniform(3,16)*SCALE), float(rng.uniform(.15,.55)), float(rng.uniform(0,2*math.pi))) for _ in range(85 if QUICK else 180)]
        self.terrain = self._terrain(340 if QUICK else 520, 180 if QUICK else 260, 2026)
        self.vignette = self._vignette()

    @staticmethod
    def _terrain(width: int, depth: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        arr = rng.normal(size=(depth,width)).astype(np.float32)
        image = Image.fromarray(np.uint8(np.clip((arr-arr.min())/max(float(np.ptp(arr)),1e-6)*255,0,255)))
        total = np.zeros((depth,width),np.float32)
        for radius,weight in ((18,1.8),(10,1.25),(5,.8),(2,.45)):
            blurred = image.filter(ImageFilter.GaussianBlur(radius))
            total += np.asarray(blurred,np.float32)/255.0*weight
        yy,xx=np.mgrid[0:depth,0:width]
        total += .45*np.sin(xx/16+np.sin(yy/22)) + .30*np.cos((xx+yy)/29)
        total -= total.min(); total /= max(float(total.max()),1e-6)
        return total

    def _vignette(self) -> np.ndarray:
        yy,xx=np.mgrid[0:H,0:W]
        nx=(xx-W/2)/(W/2); ny=(yy-H/2)/(H/2)
        return np.clip(1-.28*(nx*nx+ny*ny)**.9,.50,1.0).astype(np.float32)

    def base(self,t:float) -> Image.Image:
        arr=np.zeros((H,W,3),np.uint8)
        yy=np.linspace(0,1,H)[:,None]
        arr[...,0]=np.clip(3+yy*12,0,255)
        arr[...,1]=np.clip(5+yy*8,0,255)
        arr[...,2]=np.clip(13+yy*15,0,255)
        image=Image.fromarray(arr,"RGB").convert("RGBA")
        d=ImageDraw.Draw(image)
        for s in self.stars:
            a=int(s.a*(.75+.25*math.sin(t*1.2+s.phase)))
            d.ellipse((s.x-s.r,s.y-s.r,s.x+s.r,s.y+s.r),fill=COLORS["white"]+(a,))
        return image

    def mars_globe(self,image:Image.Image,center:Tuple[int,int],radius:int,rotation:float,alpha:int=255) -> None:
        size=radius*2
        yy,xx=np.mgrid[0:size,0:size]
        nx=(xx-radius+.5)/radius; ny=(yy-radius+.5)/radius
        rr=nx*nx+ny*ny; mask=rr<=1
        z=np.sqrt(np.clip(1-rr,0,1))
        lon=np.arctan2(nx,z)+rotation
        lat=np.arcsin(np.clip(-ny,-1,1))
        terrain=.52+.20*np.sin(lon*2.7+np.sin(lat*3.2))+.13*np.cos(lon*5.3-lat*2.1)+.08*np.sin(lon*11+lat*7)
        shade=np.clip(.12+.82*z+.20*nx,0,1)
        r=np.clip(55+165*terrain*shade,0,255)
        g=np.clip(20+78*terrain*shade,0,255)
        b=np.clip(15+48*terrain*shade,0,255)
        polar=np.clip((np.abs(lat)-1.10)/.25,0,1)
        r=r*(1-polar)+230*polar; g=g*(1-polar)+220*polar; b=b*(1-polar)+205*polar
        rgba=np.dstack([r,g,b,np.where(mask,alpha,0)]).astype(np.uint8)
        globe=Image.fromarray(rgba,"RGBA")
        gd=ImageDraw.Draw(globe)
        for cx,cy,cr,ca in self.craters:
            px=radius+cx*radius; py=radius+cy*radius
            if (cx*cx+cy*cy)<.75:
                rr2=cr*radius
                gd.ellipse((px-rr2,py-rr2,px+rr2,py+rr2),outline=(75,28,21,int(80*ca)),width=max(1,int(2*SCALE)))
                gd.arc((px-rr2*.7,py-rr2*.7,px+rr2*.7,py+rr2*.7),190,350,fill=(255,150,90,int(60*ca)),width=max(1,int(SCALE)))
        glow=Image.new("RGBA",(size,size),(0,0,0,0)); gdraw=ImageDraw.Draw(glow)
        gdraw.ellipse((5,5,size-5,size-5),outline=(255,120,70,95),width=max(2,int(4*SCALE)))
        globe.alpha_composite(glow.filter(ImageFilter.GaussianBlur(max(4,int(18*SCALE)))))
        globe.alpha_composite(glow)
        image.alpha_composite(globe,(center[0]-radius,center[1]-radius))

    def draw_orbit(self,image:Image.Image,t:float,local:float) -> None:
        cx,cy=int(W*.50),int(H*.46)
        orbit_w,orbit_h=int(W*.72),int(H*.18)
        d=ImageDraw.Draw(image)
        d.ellipse((cx-orbit_w//2,cy-orbit_h//2,cx+orbit_w//2,cy+orbit_h//2),outline=(255,205,115,90),width=max(1,int(2*SCALE)))
        sun_x=cx-orbit_w//2-int(45*SCALE); sun_y=cy
        for rr,a in ((55,22),(32,45),(15,220)):
            r=int(rr*SCALE); d.ellipse((sun_x-r,sun_y-r,sun_x+r,sun_y+r),fill=COLORS["gold"]+(a,))
        angle=2*math.pi*(.06+.88*local)
        px=cx+math.cos(angle)*orbit_w/2; py=cy+math.sin(angle)*orbit_h/2
        self.mars_globe(image,(int(px),int(py)),int(68*SCALE),-t*.8)
        labels=[("SPRING",.10),("SUMMER",.34),("AUTUMN",.60),("WINTER",.84)]
        for label,p in labels:
            a=2*math.pi*p
            x=cx+math.cos(a)*orbit_w/2; y=cy+math.sin(a)*orbit_h/2
            text(image,label,(x,y-int(45*SCALE)),13,COLORS["white"]+(190,),True,"ma",1)

    def terrain_scene(self,image:Image.Image,t:float,season_index:float=0.0) -> None:
        horizon=int(H*.47)
        d=ImageDraw.Draw(image)
        warm=math.sin(season_index*math.pi*2)*.5+.5
        for y in range(horizon):
            p=y/max(horizon,1)
            d.line((0,y,W,y),fill=(int(5+35*p+16*warm*p),int(8+17*p),int(18+20*p),255))
        # Sun height follows season; it remains visually stylized, not a solar-position calculation.
        sun_y=int(H*(.19+.11*(1-warm)))
        sun_x=int(W*(.77-.10*math.sin(t*.18)))
        haze=Image.new("RGBA",SIZE,(0,0,0,0)); hd=ImageDraw.Draw(haze)
        for rr,a in ((120,10),(70,22),(24,190)):
            r=int(rr*SCALE); hd.ellipse((sun_x-r,sun_y-r,sun_x+r,sun_y+r),fill=COLORS["gold"]+(a,))
        image.alpha_composite(haze.filter(ImageFilter.GaussianBlur(max(4,int(25*SCALE)))))
        depth,width=self.terrain.shape
        td=ImageDraw.Draw(image)
        strips=100 if QUICK else 150
        for i in range(strips):
            z=i/(strips-1)
            y=horizon+int((z**1.7)*(H-horizon))
            row=min(depth-1,int((z+t*.015)%1*depth))
            scale_x=lerp(.28,2.2,z)
            samples=max(18,int(W/(5*SCALE)))
            points=[]
            for j in range(samples+1):
                x=j/samples*W
                src=int((j/samples*width/scale_x+t*4+i*.3)%width)
                elev=float(self.terrain[row,src])
                py=y-int(elev*(18+z*90)*SCALE)
                points.append((x,py))
            base_y=min(H,y+int(25*SCALE))
            colour=(int(lerp(58,170,z)),int(lerp(27,78,z)),int(lerp(24,45,z)),255)
            td.polygon(points+[(W,base_y),(0,base_y)],fill=colour)
        # rover silhouette
        rx=int(W*.47); ry=int(H*.71)
        td.rectangle((rx-int(42*SCALE),ry-int(12*SCALE),rx+int(42*SCALE),ry+int(18*SCALE)),fill=(10,10,12,245))
        td.rectangle((rx-int(10*SCALE),ry-int(55*SCALE),rx+int(8*SCALE),ry-int(12*SCALE)),fill=(10,10,12,245))
        td.line((rx,ry-int(52*SCALE),rx+int(34*SCALE),ry-int(75*SCALE)),fill=(10,10,12,245),width=max(2,int(5*SCALE)))
        for wx in (-28,28):
            td.ellipse((rx+int((wx-12)*SCALE),ry+int(8*SCALE),rx+int((wx+12)*SCALE),ry+int(32*SCALE)),fill=(6,6,8,255))
        # drifting dust
        layer=Image.new("RGBA",SIZE,(0,0,0,0)); ld=ImageDraw.Draw(layer)
        for x,y,r,s,ph in self.dust:
            xx=(x+t*(10+22*s)*SCALE)%W; yy=(y+math.sin(t*.6+ph)*18*SCALE)%H
            if yy>horizon*.55:
                ld.ellipse((xx-r,yy-r*.22,xx+r,yy+r*.22),fill=COLORS["orange"]+(int(9+22*s),))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(2,int(8*SCALE)))))

    def title_band(self,image:Image.Image,small:str="CURIOSITY • GALE CRATER • FIRST MARTIAN YEAR") -> None:
        text(image,"THE REAL TEMPERATURE OF MARS",(W//2,int(H*.075)),28,COLORS["white"]+(250,),True,"ma",1)
        text(image,"FOR ONE YEAR",(W//2,int(H*.115)),34,COLORS["gold"]+(250,),True,"ma",1)
        text(image,small,(W//2,int(H*.155)),13,COLORS["muted"]+(220,),True,"ma",1)

    def intro(self,image:Image.Image,t:float,local:float) -> None:
        radius=int(235*SCALE*(.90+.10*local))
        self.mars_globe(image,(W//2,int(H*.43)),radius,-.5+t*.22)
        self.title_band(image)
        panel(image,(int(W*.10),int(H*.70),int(W*.90),int(H*.82)),165)
        text(image,"687 EARTH DAYS",(W//2,int(H*.745)),31,COLORS["gold"]+(245,),True,"ma",1)
        text(image,"≈ 669 MARTIAN SOLS",(W//2,int(H*.790)),17,COLORS["white"]+(225,),True,"ma",1)

    def year(self,image:Image.Image,t:float,local:float) -> None:
        self.title_band(image,"ONE ORBIT • FOUR SEASONS • ONE ROVER LOCATION")
        self.draw_orbit(image,t,local)
        panel(image,(int(W*.09),int(H*.70),int(W*.91),int(H*.83)),170)
        text(image,"A MARTIAN YEAR IS ALMOST TWICE AS LONG AS OURS",(W//2,int(H*.747)),21,COLORS["white"]+(245,),True,"ma",1)
        text(image,"Curiosity sampled every season at Gale Crater",(W//2,int(H*.792)),16,COLORS["muted"]+(225,),False,"ma",1)

    def extremes(self,image:Image.Image,t:float,local:float) -> None:
        self.terrain_scene(image,t,.22)
        self.title_band(image,"PUBLISHED CURIOSITY / REMS AIR-TEMPERATURE RANGES")
        x=int(W*.50); y0=int(H*.27); y1=int(H*.70)
        d=ImageDraw.Draw(image)
        d.rounded_rectangle((x-int(35*SCALE),y0,x+int(35*SCALE),y1),radius=int(25*SCALE),fill=(16,20,27,210),outline=(255,255,255,55),width=max(1,int(2*SCALE)))
        reveal=smoothstep(local)
        warm=float(MAX_C.max()); cold=float(MIN_C.min())
        def ty(temp:float)->int:
            return int(lerp(y0+int(25*SCALE),y1-int(25*SCALE),clamp((25-temp)/120)))
        top=ty(warm); bottom=ty(cold)
        grad=Image.new("RGBA",SIZE,(0,0,0,0)); gd=ImageDraw.Draw(grad)
        steps=max(50,int((bottom-top)*reveal))
        for i in range(steps):
            p=i/max(steps-1,1); yy=int(lerp(top,bottom,p)); col=(int(lerp(255,65,p)),int(lerp(190,126,p)),int(lerp(70,195,p)),245)
            gd.line((x-int(18*SCALE),yy,x+int(18*SCALE),yy),fill=col,width=max(1,int(2*SCALE)))
        image.alpha_composite(grad)
        text(image,f"{warm:.0f}°C",(x-int(68*SCALE),top),31,COLORS["gold"]+(250,),True,"ra",1)
        text(image,"WARMEST BIN HIGH",(x-int(68*SCALE),top+int(35*SCALE)),12,COLORS["white"]+(210,),True,"ra",1)
        text(image,f"{cold:.0f}°C",(x+int(68*SCALE),bottom),31,COLORS["ice"]+(250,),True,"la",1)
        text(image,"COLDEST BIN LOW",(x+int(68*SCALE),bottom+int(35*SCALE)),12,COLORS["white"]+(210,),True,"la",1)
        panel(image,(int(W*.10),int(H*.76),int(W*.90),int(H*.86)),175)
        text(image,"THE SAME PLANET. THE SAME CRATER. A HUGE SWING.",(W//2,int(H*.808)),20,COLORS["white"]+(245,),True,"ma",1)

    def annual(self,image:Image.Image,t:float,local:float) -> None:
        self.terrain_scene(image,t,.50)
        self.title_band(image,"12 PUBLISHED SEASONAL BINS • ONE CLEAN DATA RIBBON")
        x0,x1=int(W*.09),int(W*.91); y0,y1=int(H*.26),int(H*.69)
        panel(image,(x0,y0,x1,y1),188)
        gx0,gx1=x0+int(55*SCALE),x1-int(40*SCALE); gy0,gy1=y0+int(45*SCALE),y1-int(55*SCALE)
        d=ImageDraw.Draw(image)
        for val in (0,-20,-40,-60,-80):
            yy=int(lerp(gy0,gy1,(10-val)/100))
            d.line((gx0,yy,gx1,yy),fill=(255,255,255,35),width=max(1,int(SCALE)))
            text(image,f"{val}°",(gx0-int(12*SCALE),yy),11,COLORS["muted"]+(170,),False,"ra",1)
        reveal_count=max(2,int(math.ceil(len(BINS)*smoothstep(local))))
        xs=np.linspace(gx0,gx1,len(BINS))
        def yy(temp:float)->float: return lerp(gy0,gy1,clamp((10-temp)/100))
        # seasonal background bands, composited softly so the data stays readable.
        season_overlay=Image.new("RGBA",SIZE,(0,0,0,0)); sd=ImageDraw.Draw(season_overlay)
        for si,(name,start,end) in enumerate((("SPRING",0,3),("SUMMER",3,6),("AUTUMN",6,9),("WINTER",9,12))):
            bx0=int(lerp(gx0,gx1,start/(len(BINS)-1))); bx1=int(lerp(gx0,gx1,min(end-1,len(BINS)-1)/(len(BINS)-1)))
            sd.rectangle((bx0,gy0,bx1,gy1),fill=(255,130,65,11 if si<2 else 7))
            text(image,name,((bx0+bx1)//2,gy1+int(26*SCALE)),10,COLORS["muted"]+(190,),True,"ma",1)
        image.alpha_composite(season_overlay)
        top_pts=[]; bot_pts=[]
        for i in range(reveal_count):
            top_pts.append((float(xs[i]),float(yy(MAX_C[i])))); bot_pts.append((float(xs[i]),float(yy(MIN_C[i]))))
        ribbon=Image.new("RGBA",SIZE,(0,0,0,0)); rd=ImageDraw.Draw(ribbon)
        if len(top_pts)>=2:
            rd.polygon(top_pts+bot_pts[::-1],fill=(255,116,65,42))
            rd.line(top_pts,fill=COLORS["gold"]+(245,),width=max(2,int(5*SCALE)),joint="curve")
            rd.line(bot_pts,fill=COLORS["cyan"]+(235,),width=max(2,int(5*SCALE)),joint="curve")
            for p in top_pts: rd.ellipse((p[0]-4*SCALE,p[1]-4*SCALE,p[0]+4*SCALE,p[1]+4*SCALE),fill=COLORS["gold"]+(245,))
            for p in bot_pts: rd.ellipse((p[0]-4*SCALE,p[1]-4*SCALE,p[0]+4*SCALE,p[1]+4*SCALE),fill=COLORS["cyan"]+(235,))
        image.alpha_composite(ribbon)
        text(image,"AFTERNOON HIGH",(gx0,gy0-int(17*SCALE)),12,COLORS["gold"]+(230,),True,"la",1)
        text(image,"OVERNIGHT LOW",(gx1,gy0-int(17*SCALE)),12,COLORS["cyan"]+(230,),True,"ra",1)
        panel(image,(int(W*.10),int(H*.74),int(W*.90),int(H*.85)),170)
        text(image,f"LARGEST PUBLISHED BIN RANGE  ≈  {RANGE_C.max():.0f}°C",(W//2,int(H*.792)),21,COLORS["white"]+(245,),True,"ma",1)

    def seasons(self,image:Image.Image,t:float,local:float) -> None:
        season_pos=local
        self.terrain_scene(image,t,season_pos)
        self.title_band(image,"THE SEASONAL STORY • GALE CRATER")
        labels=[("SPRING",0.00,0.25),("SUMMER",0.25,0.50),("AUTUMN",0.50,0.75),("WINTER",0.75,1.01)]
        current=labels[-1][0]
        for name,a,b in labels:
            if a<=season_pos<b: current=name
        panel(image,(int(W*.12),int(H*.23),int(W*.88),int(H*.38)),160)
        text(image,current,(W//2,int(H*.292)),40,COLORS["gold"]+(250,),True,"ma",1)
        if current in ("SPRING","SUMMER"):
            note="AFTERNOONS APPROACH 0°C • NIGHTS STAY FAR BELOW FREEZING"
        else:
            note="AFTERNOONS FALL • WINTER NIGHTS APPROACH −81°C"
        text(image,note,(W//2,int(H*.345)),14,COLORS["white"]+(225,),True,"ma",1)
        # one-year progress rail
        x0,x1=int(W*.13),int(W*.87); y=int(H*.80)
        d=ImageDraw.Draw(image); d.line((x0,y,x1,y),fill=(255,255,255,70),width=max(2,int(4*SCALE)))
        fill_x=int(lerp(x0,x1,local)); d.line((x0,y,fill_x,y),fill=COLORS["orange"]+(245,),width=max(3,int(8*SCALE)))
        sol=int(lerp(0,669,local)); text(image,f"SOL {sol}",(fill_x,y-int(25*SCALE)),14,COLORS["white"]+(240,),True,"ma",1)

    def finale(self,image:Image.Image,t:float,local:float) -> None:
        self.terrain_scene(image,t,.84)
        overlay=Image.new("RGBA",SIZE,(0,0,0,0)); od=ImageDraw.Draw(overlay)
        od.rectangle((0,0,W,H),fill=(3,4,8,int(70+80*local)))
        image.alpha_composite(overlay)
        self.title_band(image,"REAL ROVER AIR TEMPERATURE • ONE LOCATION • ONE MARTIAN YEAR")
        panel(image,(int(W*.08),int(H*.24),int(W*.92),int(H*.69)),195)
        text(image,"GALE CRATER",(W//2,int(H*.31)),23,COLORS["muted"]+(230,),True,"ma",1)
        text(image,f"{MAX_C.max():.0f}°C",(int(W*.31),int(H*.43)),48,COLORS["gold"]+(250,),True,"ma",1)
        text(image,"WARMEST BIN HIGH",(int(W*.31),int(H*.49)),12,COLORS["white"]+(210,),True,"ma",1)
        text(image,f"{MIN_C.min():.0f}°C",(int(W*.69),int(H*.43)),48,COLORS["ice"]+(250,),True,"ma",1)
        text(image,"COLDEST BIN LOW",(int(W*.69),int(H*.49)),12,COLORS["white"]+(210,),True,"ma",1)
        d=ImageDraw.Draw(image); d.line((W//2,int(H*.36),W//2,int(H*.56)),fill=(255,255,255,45),width=max(1,int(2*SCALE)))
        wrapped(image,"Mars has no single temperature. These are real seasonal air-temperature ranges measured by Curiosity at Gale Crater.",(int(W*.15),int(H*.58),int(W*.85),int(H*.67)),17,COLORS["white"]+(235,),False,6,"center")
        text(image,"SOURCE  NASA/JPL-CALTECH/CAB(CSIC-INTA) • REMS",(W//2,int(H*.78)),12,COLORS["muted"]+(220,),True,"ma",1)

    def hud(self,image:Image.Image,t:float) -> None:
        # top/bottom safe-zone accents
        d=ImageDraw.Draw(image)
        m=int(36*SCALE); l=int(45*SCALE)
        for x,y,sx,sy in ((m,m,1,1),(W-m,m,-1,1),(m,H-m,1,-1),(W-m,H-m,-1,-1)):
            d.line((x,y,x+sx*l,y),fill=(255,180,100,65),width=max(1,int(SCALE)))
            d.line((x,y,x,y+sy*l),fill=(255,180,100,65),width=max(1,int(SCALE)))
        cap=caption_at(t)
        if cap:
            panel(image,(int(W*.06),int(H*.865),int(W*.94),int(H*.965)),155,18)
            wrapped(image,cap,(int(W*.10),int(H*.886),int(W*.90),int(H*.952)),15,COLORS["white"]+(245,),False,4,"center")

    def render(self,t:float) -> np.ndarray:
        image=self.base(t)
        name,a,b=shot_at(t); local=smoothstep((t-a)/max(b-a,1e-9))
        if name=="intro": self.intro(image,t,local)
        elif name=="year": self.year(image,t,local)
        elif name=="extremes": self.extremes(image,t,local)
        elif name=="annual": self.annual(image,t,local)
        elif name=="seasons": self.seasons(image,t,local)
        else: self.finale(image,t,local)
        self.hud(image,t)
        arr=np.asarray(image.convert("RGB"),dtype=np.float32)
        arr*=self.vignette[...,None]
        arr=np.clip(arr,0,255).astype(np.uint8)
        arr=np.asarray(ImageEnhance.Contrast(Image.fromarray(arr)).enhance(1.06))
        return arr


# -----------------------------------------------------------------------------
# Original procedural soundtrack
# -----------------------------------------------------------------------------


def envelope(n:int,attack:float=.03,release:float=.12) -> np.ndarray:
    x=np.ones(n,np.float32)
    a=max(1,int(n*attack)); r=max(1,int(n*release))
    x[:a]=np.linspace(0,1,a); x[-r:]=np.linspace(1,0,r)
    return x


def add_tone(track:np.ndarray,sr:int,start:float,dur:float,freq:float,amp:float,pan:float=0.0,kind:str="sine") -> None:
    i0=max(0,int(start*sr)); n=min(len(track)-i0,int(dur*sr))
    if n<=0:return
    tt=np.arange(n,dtype=np.float32)/sr
    if kind=="triangle":
        wavef=(2/np.pi)*np.arcsin(np.sin(2*np.pi*freq*tt))
    else:
        wavef=np.sin(2*np.pi*freq*tt)
    wavef*=envelope(n)
    left=math.sqrt((1-pan)/2); right=math.sqrt((1+pan)/2)
    track[i0:i0+n,0]+=wavef*amp*left; track[i0:i0+n,1]+=wavef*amp*right


def add_noise(track:np.ndarray,sr:int,start:float,dur:float,amp:float,pan:float=0.0,seed:int=0,lowpass:int=70) -> None:
    i0=max(0,int(start*sr)); n=min(len(track)-i0,int(dur*sr))
    if n<=0:return
    rng=np.random.default_rng(seed)
    noise=rng.normal(0,1,n).astype(np.float32)
    kernel=np.ones(max(2,lowpass),np.float32)/max(2,lowpass)
    noise=np.convolve(noise,kernel,mode="same")
    noise/=max(float(np.max(np.abs(noise))),1e-6); noise*=envelope(n,.08,.20)
    left=math.sqrt((1-pan)/2); right=math.sqrt((1+pan)/2)
    track[i0:i0+n,0]+=noise*amp*left; track[i0:i0+n,1]+=noise*amp*right


def make_soundtrack(path:Path) -> Path:
    sr=44100; n=int(DURATION*sr); track=np.zeros((n,2),np.float32)
    # deep atmospheric bed
    for f,a,p in ((43,.12,-.15),(65,.08,.18),(86,.045,0)):
        add_tone(track,sr,0,DURATION,f,a,p)
    # wind and dust texture
    add_noise(track,sr,0,DURATION,.075,0,2026,180)
    add_noise(track,sr,0,DURATION,.035,-.35,2027,28)
    # transitions
    for idx,(_,start,_) in enumerate(SHOT_PLAN[1:],1):
        add_noise(track,sr,max(0,start-.38),.75,.13,.45 if idx%2 else -.45,100+idx,8)
        add_tone(track,sr,start,.45,55+idx*8,.15,0,"triangle")
    # pulse clock during annual data scene
    annual=next(s for s in SHOT_PLAN if s[0]=="annual")
    beat=.72 if not QUICK else .22
    tt=annual[1]
    k=0
    while tt<annual[2]:
        add_tone(track,sr,tt,.16,110 if k%2==0 else 82,.08,-.22 if k%2==0 else .22,"triangle")
        tt+=beat; k+=1
    # final shimmer
    finale=SHOT_PLAN[-1][1]
    for i,f in enumerate((220,330,440,660)):
        add_tone(track,sr,finale+i*.10,DURATION-finale-.1*i,f,.035,(-.5+i/3))
    peak=max(float(np.max(np.abs(track))),1e-6)
    track=np.tanh(track/(peak*.78))*0.88
    pcm=np.int16(np.clip(track,-1,1)*32767)
    with wave.open(str(path),"wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(sr); wav.writeframes(pcm.tobytes())
    return path


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------


def mux_audio(video:Path,audio:Path,output:Path) -> Path:
    cmd=["ffmpeg","-y","-loglevel","error","-i",str(video),"-i",str(audio),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart",str(output)]
    subprocess.run(cmd,check=True)
    return output


def contact_sheet(video:Path,path:Path) -> Path:
    reader=iio.get_reader(video)
    frames=[]
    for p in np.linspace(.05,.94,6):
        frame=reader.get_data(int(p*DURATION*FPS))
        frames.append(Image.fromarray(frame).resize((270,480),Image.Resampling.LANCZOS))
    reader.close()
    sheet=Image.new("RGB",(270*3,480*2),(8,8,10))
    for i,frame in enumerate(frames): sheet.paste(frame,((i%3)*270,(i//3)*480))
    sheet.save(path,quality=92)
    return path



