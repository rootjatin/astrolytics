
from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

Vec2 = Tuple[float, float]

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, float(x)))

def lerp(a, b, t):
    return a + (b - a) * t

def ease(t):
    t = clamp(t)
    return 0.5 - 0.5 * math.cos(math.pi * t)

def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)

def fract(x):
    return x - math.floor(x)

def hash01(n):
    return fract(math.sin(n * 12.9898 + 78.233) * 43758.5453)

def norm(vx, vy):
    d = math.hypot(vx, vy)
    if d == 0:
        return 0.0, 0.0
    return vx / d, vy / d

def font(size: int, bold: bool = False):
    cands = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

@dataclass
class Shot:
    name: str
    start: float
    end: float
    caption: str

@dataclass
class Spec:
    title: str
    subtitle: str
    basename: str
    shots: List[Shot]
    draw: Callable
    notes: List[str]

class Renderer:
    def __init__(self, spec: Spec, quick: bool = False, fourk: bool = False):
        self.spec = spec
        self.quick = quick
        self.fourk = fourk and not quick
        self.W = 540 if quick else (2160 if self.fourk else 1080)
        self.H = 960 if quick else (3840 if self.fourk else 1920)
        self.FPS = 6 if quick else (30 if self.fourk else 24)
        self.duration = 13.0 if quick else spec.shots[-1].end
        self.S = self.W / 1080.0
        self.root = Path(f'{spec.basename}_output')
        self.preview = self.root / 'previews'
        self.data = self.root / 'data'
        self.preview.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(abs(hash(spec.basename)) % (2**32))
        self.starfield = self._make_starfield(950 if not quick else 280)
        self.vignette = self._make_vignette()
        self.particle_bank = self._make_particles(900 if not quick else 260)
        self.asteroids = self._make_asteroids(240 if not quick else 100)

    def _make_vignette(self):
        y, x = np.ogrid[:self.H, :self.W]
        nx = (x - self.W / 2) / (self.W / 2)
        ny = (y - self.H / 2) / (self.H / 2)
        r = np.sqrt(nx * nx + ny * ny)
        return np.clip(1 - 0.34 * np.maximum(0, r - 0.15) ** 1.6, 0.50, 1.0).astype(np.float32)

    def _make_starfield(self, n):
        pts = []
        for i in range(n):
            x = float(self.rng.random())
            y = float(self.rng.random())
            b = float(self.rng.random() ** 3)
            r = 0.4 + 2.4 * b
            pts.append((x, y, b, r, hash01(i + 17)))
        return pts

    def _make_particles(self, n):
        pts = []
        for i in range(n):
            a = float(self.rng.random() * math.tau)
            rr = float(0.18 + self.rng.random() * 0.82)
            z = float(self.rng.normal(0.0, 0.06))
            size = float(0.5 + self.rng.random() * 2.2)
            phase = float(self.rng.random() * math.tau)
            pts.append((a, rr, z, size, phase))
        return pts

    def _make_asteroids(self, n):
        pts = []
        for i in range(n):
            a = float(self.rng.random() * math.tau)
            rr = float(0.38 + self.rng.random() * 0.38)
            e = float(self.rng.random() * 0.08)
            size = float(1.5 + self.rng.random() * 5.0)
            hue = float(self.rng.random())
            pts.append((a, rr, e, size, hue))
        return pts

    def shot_at(self, t):
        if self.quick:
            q = t / 13.0
            idx = min(len(self.spec.shots) - 1, int(q * len(self.spec.shots)))
            sh = self.spec.shots[idx]
            p = q * len(self.spec.shots) - idx
            return sh, clamp(p)
        for sh in self.spec.shots:
            if sh.start <= t < sh.end:
                return sh, clamp((t - sh.start) / (sh.end - sh.start))
        sh = self.spec.shots[-1]
        return sh, 1.0

    def base(self, t, nebula=True):
        img = Image.new('RGBA', (self.W, self.H), (2, 5, 16, 255))
        d = ImageDraw.Draw(img, 'RGBA')
        if nebula:
            neb = Image.new('RGBA', (self.W, self.H), (0, 0, 0, 0))
            nd = ImageDraw.Draw(neb, 'RGBA')
            for k in range(10):
                cx = self.W * (0.06 + 0.10 * (k % 8)) + math.sin(t * 0.04 + k) * self.W * 0.09
                cy = self.H * (0.16 + 0.09 * ((k * 3) % 7)) + math.cos(t * 0.05 + k * 0.7) * self.H * 0.05
                rr = self.W * (0.11 + 0.05 * (k % 3))
                col = ((20, 38, 94, 26), (80, 24, 102, 22), (18, 94, 120, 20), (22, 64, 40, 18))[k % 4]
                nd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=col)
            neb = neb.filter(ImageFilter.GaussianBlur(max(10, int(95 * self.S))))
            img.alpha_composite(neb)
            d = ImageDraw.Draw(img, 'RGBA')
        drift = t * 0.002
        for i, (x, y, b, r, h) in enumerate(self.starfield):
            xx = ((x + drift * (0.2 + h)) % 1.0) * self.W
            yy = y * self.H
            tw = 0.55 + 0.45 * math.sin(t * (1.2 + 2.1 * h) + i)
            a = int(50 + 205 * b * tw)
            rad = max(0.4 * self.S, r * self.S)
            c = (205 + int(45 * h), 220 + int(25 * h), 255, a)
            d.ellipse((xx - rad, yy - rad, xx + rad, yy + rad), fill=c)
        return img

    def glow_circle(self, img, xy, r, color, core=(255, 255, 255, 255), blur=26):
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, 'RGBA')
        x, y = xy
        for m, a in [(2.2, 18), (1.7, 30), (1.35, 55), (1.08, 82)]:
            rr = r * m
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(*color[:3], a))
        layer = layer.filter(ImageFilter.GaussianBlur(max(2, int(blur * self.S))))
        img.alpha_composite(layer)
        d = ImageDraw.Draw(img, 'RGBA')
        d.ellipse((x - r, y - r, x + r, y + r), fill=core)

    def orbit(self, img, center, rx, ry, alpha=80, width=2, dash=False):
        d = ImageDraw.Draw(img, 'RGBA')
        x, y = center
        if not dash:
            d.ellipse((x - rx, y - ry, x + rx, y + ry), outline=(118, 177, 255, alpha), width=max(1, int(width * self.S)))
            return
        steps = 140
        pts = []
        for i in range(steps + 1):
            a = i / steps * math.tau
            pts.append((x + math.cos(a) * rx, y + math.sin(a) * ry))
        for i in range(0, steps, 6):
            if (i // 6) % 2 == 0:
                d.line(pts[i:i+6], fill=(118, 177, 255, alpha), width=max(1, int(width * self.S)))

    def trail(self, img, points, color=(100, 220, 255), width=5):
        if len(points) < 2:
            return
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, 'RGBA')
        n = len(points)
        for i in range(1, n):
            a = int(220 * i / n)
            d.line([points[i - 1], points[i]], fill=(*color[:3], a), width=max(1, int(width * self.S)))
        layer = layer.filter(ImageFilter.GaussianBlur(max(1, int(2 * self.S))))
        img.alpha_composite(layer)

    def planet(self, img, xy, r, base=(40, 116, 204), light=(-0.42, -0.28), rim=(130, 220, 255), bands=False):
        x, y = xy
        r = int(r)
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        mask = xx * xx + yy * yy <= r * r
        nx = xx / max(1, r)
        ny = yy / max(1, r)
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        lx, ly = light
        lz = max(0.2, math.sqrt(max(0.0, 1 - lx * lx - ly * ly)))
        lam = np.clip(nx * lx + ny * ly + nz * lz, -0.2, 1)
        shade = np.clip(0.18 + 0.88 * np.maximum(lam, 0), 0, 1)
        arr = np.zeros((2 * r + 1, 2 * r + 1, 4), dtype=np.uint8)
        for k, c in enumerate(base):
            arr[..., k] = (c * shade).astype(np.uint8)
        arr[..., 3] = (mask * 255).astype(np.uint8)
        tex = (np.sin(xx * 0.08) + np.sin(yy * 0.06 + xx * 0.03)) * 0.5
        if bands:
            tex += np.sin(yy * 0.22) * 0.8
        arr[..., 0] = np.where(mask, np.clip(arr[..., 0] + tex * 10, 0, 255), 0)
        arr[..., 1] = np.where(mask, np.clip(arr[..., 1] + tex * 12, 0, 255), 0)
        arr[..., 2] = np.where(mask, np.clip(arr[..., 2] + tex * 8, 0, 255), 0)
        p = Image.fromarray(arr, 'RGBA')
        img.alpha_composite(p, (int(x - r), int(y - r)))
        d = ImageDraw.Draw(img, 'RGBA')
        d.arc((x - r, y - r, x + r, y + r), 100, 258, fill=(*rim[:3], 120), width=max(1, int(3 * self.S)))

    def ring_system(self, img, center, rx, ry, density=1.0, tilt=0.0, gap=False, particle=False, highlight=True):
        x0, y0 = center
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, 'RGBA')
        bands = 12
        for i in range(bands):
            frac = i / max(1, bands - 1)
            rrx = rx * (0.68 + frac * 0.44)
            rry = ry * (0.68 + frac * 0.44)
            a = int(12 + 28 * (1 - abs(frac - 0.45)))
            col = (210 + int(20 * frac), 205 + int(22 * frac), 190 + int(18 * frac), int(a * density * 2.6))
            d.ellipse((x0 - rrx, y0 - rry, x0 + rrx, y0 + rry), outline=col, width=max(1, int(2 * self.S)))
        if gap:
            d.ellipse((x0 - rx * 0.96, y0 - ry * 0.96, x0 + rx * 0.96, y0 + ry * 0.96), outline=(20, 20, 24, 120), width=max(1, int(7 * self.S)))
        if particle:
            pd = ImageDraw.Draw(layer, 'RGBA')
            for i, (a, rr, z, size, phase) in enumerate(self.particle_bank):
                ang = a + tilt * 0.25 + phase * 0.1
                px = x0 + math.cos(ang) * rx * rr
                py = y0 + math.sin(ang) * ry * rr + z * ry * 0.26
                rad = max(0.7, size * self.S * (0.85 + 0.25 * math.sin(phase + tilt + rr * 8)))
                aa = int(60 + 100 * (1 - rr))
                pd.ellipse((px - rad, py - rad, px + rad, py + rad), fill=(235, 230, 214, aa))
        layer = layer.filter(ImageFilter.GaussianBlur(max(1, int(1.5 * self.S))))
        img.alpha_composite(layer)
        if highlight:
            d = ImageDraw.Draw(img, 'RGBA')
            d.arc((x0 - rx * 1.12, y0 - ry * 1.12, x0 + rx * 1.12, y0 + ry * 1.12), 0, 180, fill=(245, 238, 225, 150), width=max(1, int(6 * self.S)))

    def draw_grid(self, img, x, y, w, h, rows=4, cols=4, alpha=70):
        d = ImageDraw.Draw(img, 'RGBA')
        d.rounded_rectangle((x, y, x+w, y+h), radius=int(18*self.S), outline=(90,180,255,alpha), width=max(1, int(2*self.S)))
        for i in range(1, cols):
            xx = x + w * i / cols
            d.line((xx, y, xx, y+h), fill=(90,180,255,alpha//2), width=max(1, int(1*self.S)))
        for j in range(1, rows):
            yy = y + h * j / rows
            d.line((x, yy, x+w, yy), fill=(90,180,255,alpha//2), width=max(1, int(1*self.S)))

    def radar(self, img, center, radius, spokes=6, color=(85,228,255)):
        d = ImageDraw.Draw(img, 'RGBA')
        x, y = center
        for j in range(1, 5):
            rr = radius * j / 4
            d.ellipse((x-rr, y-rr, x+rr, y+rr), outline=(*color, 35), width=max(1, int(1*self.S)))
        for i in range(spokes):
            a = i / spokes * math.tau
            d.line((x, y, x + math.cos(a)*radius, y + math.sin(a)*radius), fill=(*color, 45), width=max(1, int(1*self.S)))

    def arrow(self, img, a, b, color=(255,255,255), width=4):
        d = ImageDraw.Draw(img, 'RGBA')
        d.line([a,b], fill=(*color[:3],220), width=max(1,int(width*self.S)))
        dx, dy = b[0]-a[0], b[1]-a[1]
        ux, uy = norm(dx, dy)
        px, py = -uy, ux
        ah = 16*self.S
        p1 = (b[0]-ux*ah+px*ah*0.5, b[1]-uy*ah+py*ah*0.5)
        p2 = (b[0]-ux*ah-px*ah*0.5, b[1]-uy*ah-py*ah*0.5)
        d.polygon([b,p1,p2], fill=(*color[:3],220))

    def text_center(self, img, text, y, size=54, fill=(245,250,255,255), bold=True, stroke=3):
        d = ImageDraw.Draw(img, 'RGBA')
        f = font(max(9, int(size * self.S)), bold)
        bb = d.textbbox((0, 0), text, font=f, stroke_width=max(0, int(stroke * self.S)))
        x = (self.W - (bb[2] - bb[0])) / 2
        d.text((x, y), text, font=f, fill=fill, stroke_width=max(0, int(stroke * self.S)), stroke_fill=(0, 0, 0, 180))

    def small_label(self, img, text, x, y, color=(85,228,255), fill=(3,10,24,185)):
        d = ImageDraw.Draw(img, 'RGBA')
        f = font(max(9, int(20*self.S)), True)
        bb = d.textbbox((0,0), text, font=f)
        w = bb[2]-bb[0] + int(26*self.S)
        h = bb[3]-bb[1] + int(16*self.S)
        d.rounded_rectangle((x, y, x+w, y+h), radius=int(14*self.S), fill=fill, outline=(*color, 90), width=max(1, int(2*self.S)))
        d.text((x+13*self.S, y+8*self.S), text, font=f, fill=(*color,255))

    def caption(self, img, text, progress):
        if not text:
            return
        d = ImageDraw.Draw(img, 'RGBA')
        maxw = int(self.W * 0.84)
        fs = max(16, int(37 * self.S))
        f = font(fs, True)
        words = text.split()
        lines = []
        cur = ''
        for w in words:
            test = (cur + ' ' + w).strip()
            bb = d.textbbox((0, 0), test, font=f)
            if bb[2] - bb[0] > maxw and cur:
                lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)
        lines = lines[:4]
        lh = int(fs * 1.20)
        h = lh * len(lines) + int(34 * self.S)
        y = self.H - h - int(82 * self.S)
        box = (int(self.W * 0.06), y, int(self.W * 0.94), y + h)
        panel = Image.new('RGBA', img.size, (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel, 'RGBA')
        pd.rounded_rectangle(box, radius=int(24 * self.S), fill=(2, 8, 20, 186), outline=(120, 200, 255, 50), width=max(1, int(2 * self.S)))
        img.alpha_composite(panel)
        pd = ImageDraw.Draw(img, 'RGBA')
        pd.rounded_rectangle((box[0], box[1], box[0] + (box[2] - box[0]) * clamp(progress), box[1] + max(2, int(4 * self.S))), radius=2, fill=(83, 225, 255, 210))
        yy = y + int(14 * self.S)
        for line in lines:
            bb = d.textbbox((0, 0), line, font=f)
            x = (self.W - (bb[2] - bb[0])) / 2
            d.text((x, yy), line, font=f, fill=(245, 249, 255, 255), stroke_width=max(1, int(2 * self.S)), stroke_fill=(0, 0, 0, 180))
            yy += lh

    def hud(self, img, label, value, y, color=(85, 228, 255)):
        d = ImageDraw.Draw(img, 'RGBA')
        x = int(68 * self.S)
        w = int(420 * self.S)
        h = int(62 * self.S)
        d.rounded_rectangle((x, y, x + w, y + h), radius=int(16 * self.S), fill=(3, 10, 24, 185), outline=(*color, 85), width=max(1, int(2 * self.S)))
        d.text((x + 16 * self.S, y + 10 * self.S), label, font=font(max(9, int(19 * self.S)), True), fill=(*color, 255))
        d.text((x + w - 16 * self.S, y + 9 * self.S), value, font=font(max(9, int(24 * self.S)), True), fill=(248, 251, 255, 255), anchor='ra')

    def meter(self, img, x, y, w, value, label, color=(255,180,110)):
        d = ImageDraw.Draw(img, 'RGBA')
        h = int(18*self.S)
        d.rounded_rectangle((x,y,x+w,y+h), radius=int(10*self.S), fill=(15,26,46,190), outline=(100,150,210,90), width=max(1,int(2*self.S)))
        d.rounded_rectangle((x,y,x+w*clamp(value),y+h), radius=int(10*self.S), fill=(*color,210))
        d.text((x, y-int(30*self.S)), label, font=font(max(9, int(19*self.S)), True), fill=(235,242,255,255))

    def transition(self, img, p):
        q = min(p, 1-p)
        if q < 0.035:
            a = int(72 * (1 - q / 0.035))
            img.alpha_composite(Image.new('RGBA', img.size, (138, 214, 255, a)))

    def grade(self, img):
        arr = np.asarray(img.convert('RGB')).astype(np.float32)
        arr *= self.vignette[..., None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        out = ImageEnhance.Contrast(out).enhance(1.13)
        out = ImageEnhance.Color(out).enhance(1.10)
        return out

    def render(self, t):
        sh, p = self.shot_at(t)
        img = self.base(t)
        self.spec.draw(self, img, t, sh, p)
        if p < 0.14:
            self.text_center(img, sh.name.upper().replace('_', ' '), int(90 * self.S), size=28, fill=(145, 216, 255, 230), bold=True, stroke=2)
        self.caption(img, sh.caption, p)
        self.transition(img, p)
        return np.asarray(self.grade(img))

    def srt(self):
        def ts(x):
            h = int(x // 3600)
            x -= h * 3600
            m = int(x // 60)
            x -= m * 60
            s = int(x)
            ms = int(round((x - s) * 1000))
            if ms == 1000:
                s += 1
                ms = 0
            return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'
        out = []
        if self.quick:
            dur = 13 / len(self.spec.shots)
            for i, sh in enumerate(self.spec.shots, 1):
                out += [str(i), f'{ts((i-1)*dur)} --> {ts(i*dur - .03)}', sh.caption, '']
        else:
            for i, sh in enumerate(self.spec.shots, 1):
                out += [str(i), f'{ts(sh.start + .2)} --> {ts(sh.end - .08)}', sh.caption, '']
        p = self.root / f'{self.spec.basename}.srt'
        p.write_text('\n'.join(out), encoding='utf-8')
        return p

    def previews(self):
        paths = []
        for i, sh in enumerate(self.spec.shots, 1):
            t = (i - .5) * 13 / len(self.spec.shots) if self.quick else (sh.start + sh.end) / 2
            p = self.preview / f'preview_{i:02d}_{sh.name}.jpg'
            Image.fromarray(self.render(t)).save(p, quality=92)
            paths.append(p)
        tw = 270 if not self.quick else 180
        th = int(tw * 16 / 9)
        cols = 3
        rows = math.ceil(len(paths) / cols)
        sheet = Image.new('RGB', (tw * cols, th * rows), (3, 6, 14))
        for i, p in enumerate(paths):
            im = Image.open(p).convert('RGB')
            im.thumbnail((tw, th), Image.Resampling.LANCZOS)
            c = Image.new('RGB', (tw, th), (3, 6, 14))
            c.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
            sheet.paste(c, ((i % cols) * tw, (i // cols) * th))
        sp = self.preview / f'{self.spec.basename}_contact_sheet.jpg'
        sheet.save(sp, quality=92)
        return paths, sp

    def video(self):
        out = self.root / f"{self.spec.basename}{'_quick_preview' if self.quick else ''}.mp4"
        total = max(1, int(round(self.duration * self.FPS)))
        wr = iio.get_writer(out, fps=self.FPS, codec='libx264', quality=8, macro_block_size=None, output_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        try:
            for i in range(total):
                wr.append_data(self.render(i / self.FPS))
        finally:
            wr.close()
        return out

    def metadata(self):
        p = self.data / f'{self.spec.basename}_metadata.json'
        p.write_text(json.dumps({
            'title': self.spec.title,
            'subtitle': self.spec.subtitle,
            'basename': self.spec.basename,
            'quick': self.quick,
            'fourk': self.fourk,
            'size': [self.W, self.H],
            'fps': self.FPS,
            'duration': self.duration,
            'shots': [s.__dict__ for s in self.spec.shots],
            'notes': self.spec.notes,
        }, indent=2), encoding='utf-8')
        return p

    def readme(self):
        p = self.root / 'README.txt'
        text = [
            self.spec.title,
            '=' * len(self.spec.title),
            '',
            self.spec.subtitle,
            '',
            'Render modes',
            '------------',
            'python ' + self.spec.basename + '.py',
            'python ' + self.spec.basename + '.py --quick',
            'python ' + self.spec.basename + '.py --quick --preview-only',
            'python ' + self.spec.basename + '.py --4k',
            '',
            'Scientific / creative notes',
            '----------------------------',
        ] + ['- ' + n for n in self.spec.notes]
        p.write_text('\n'.join(text), encoding='utf-8')
        return p


def run_cli(spec: Spec):
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--4k', action='store_true')
    ap.add_argument('--preview-only', action='store_true')
    args = ap.parse_args()
    r = Renderer(spec, quick=args.quick, fourk=args.__dict__['4k'])
    r.metadata()
    r.srt()
    r.readme()
    r.previews()
    if not args.preview_only:
        r.video()
