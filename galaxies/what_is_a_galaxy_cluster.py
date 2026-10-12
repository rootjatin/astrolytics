from __future__ import annotations

"""
Cinematic vertical YouTube Short: WHAT IS A GALAXY CLUSTER?

# output : https://www.youtube.com/shorts/x6dHHcdsAIE

Schematic science visualization: one galaxy -> cluster scale -> gravity -> hot gas
-> dark matter -> gravitational lensing -> cosmic web.

Install:
    pip install numpy pillow imageio imageio-ffmpeg tqdm
Quick:
    GALAXY_CLUSTER_SHORT_QUICK=1 python what_is_a_galaxy_cluster.py
"""

import json, math, os, shutil, subprocess, wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm
try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

QUICK = os.environ.get("GALAXY_CLUSTER_SHORT_QUICK", "0") == "1"
ROOT = Path("what_is_a_galaxy_cluster_output")
PREV = ROOT / "previews"
DATA = ROOT / "data"
for p in (ROOT, PREV, DATA): p.mkdir(parents=True, exist_ok=True)

CFG: Dict[str, Any] = {
    "w": 540 if QUICK else 1080,
    "h": 960 if QUICK else 1920,
    "fps": 6 if QUICK else 24,
    "duration": 12.0 if QUICK else 58.0,
    "title": "WHAT IS A GALAXY CLUSTER?",
    "basename": "what_is_a_galaxy_cluster",
    "sr": 22050 if QUICK else 44100,
}
W, H, FPS, DUR = int(CFG["w"]), int(CFG["h"]), int(CFG["fps"]), float(CFG["duration"])
S = W / 1080.0
C = {
    "space": (2,4,13), "space2": (5,8,22), "white": (247,250,255),
    "muted": (165,187,215), "cyan": (87,221,255), "blue": (92,145,255),
    "violet": (181,111,255), "gold": (255,205,112), "gas": (119,191,255),
    "grid": (94,127,176),
}

FULL_SHOTS = [
    ("one",0,7), ("scale",7,17), ("gravity",17,27), ("gas",27,37),
    ("dark",37,46), ("lens",46,53), ("web",53,58),
]
FULL_CAPS = [
    (0.5,6.6,"A galaxy is already enormous. A galaxy cluster is hundreds to thousands of galaxies gathered into one gravitational system."),
    (7.3,16.6,"The galaxies can span millions of light-years, moving through a shared gravitational potential."),
    (17.3,26.6,"Gravity binds the cluster together. The visible galaxies are only part of what is there."),
    (27.3,36.6,"Between the galaxies is extremely hot, thin gas. It shines strongly in X-rays and can reach tens of millions of degrees."),
    (37.3,45.6,"Most of the cluster's gravitating mass is invisible dark matter, mapped through its effects on motion and light."),
    (46.3,52.6,"A massive cluster bends spacetime enough to stretch and magnify more distant galaxies behind it."),
    (53.1,57.7,"Galaxy clusters sit at dense intersections of the cosmic web: some of the largest structures held together by gravity."),
]
scale_t = DUR/58.0 if QUICK else 1.0
SHOTS = [(n,a*scale_t,b*scale_t) for n,a,b in FULL_SHOTS]
CAPS = [(a*scale_t,b*scale_t,t) for a,b,t in FULL_CAPS]

def clamp(x,a=0.0,b=1.0): return max(a,min(b,float(x)))
def smooth(x):
    x=clamp(x); return x*x*(3-2*x)
def smoother(x):
    x=clamp(x); return x*x*x*(x*(x*6-15)+10)
def lerp(a,b,t): return a+(b-a)*t

def shot_at(t):
    for n,a,b in SHOTS:
        if a <= t < b: return n,a,b
    return SHOTS[-1]

def cap_at(t):
    """Return a short readable page of the active caption."""
    for start, end, value in CAPS:
        if start <= t < end:
            words = str(value).split()
            if not words:
                return None
            target_words = 18
            page_count = max(1, (len(words) + target_words - 1) // target_words)
            base = len(words) // page_count
            extra = len(words) % page_count
            progress = (t - start) / max(end - start, 1e-9)
            page = min(page_count - 1, max(0, int(progress * page_count)))
            first = page * base + min(page, extra)
            count = base + (1 if page < extra else 0)
            return " ".join(words[first:first + count])
    return None

def font(size,bold=False,condensed=False):
    candidates=[]
    if condensed and bold: candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf","DejaVuSansCondensed-Bold.ttf"]
    if condensed: candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf","DejaVuSansCondensed.ttf"]
    candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for f in candidates:
        try: return ImageFont.truetype(f,size=size)
        except Exception: pass
    return ImageFont.load_default()

def text(img,s,xy,size,fill,bold=False,condensed=False,anchor="la",stroke=2):
    ImageDraw.Draw(img).text(xy,s,font=font(size,bold,condensed),fill=fill,anchor=anchor,stroke_width=stroke,stroke_fill=(0,0,0,min(230,fill[3])))

def wrapped(img,s,xy,maxw,size,fill,spacing=4):
    d=ImageDraw.Draw(img); f=font(size); lines=[]; cur=""
    for word in s.split():
        q=word if not cur else cur+" "+word
        if d.textbbox((0,0),q,font=f,stroke_width=1)[2] <= maxw: cur=q
        else:
            if cur: lines.append(cur)
            cur=word
    if cur: lines.append(cur)
    x,y=xy
    for line in lines[:4]:
        d.text((x,y),line,font=f,fill=fill,stroke_width=2,stroke_fill=(0,0,0,220))
        y += d.textbbox((x,y),line,font=f,stroke_width=2)[3]-y + spacing

def srt_time(sec):
    ms=int(round(sec*1000)); h=ms//3600000; ms%=3600000; m=ms//60000; ms%=60000; s=ms//1000; ms%=1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(path):
    out=[]
    for i,(a,b,s) in enumerate(CAPS,1): out += [str(i),f"{srt_time(a)} --> {srt_time(b)}",s,""]
    path.write_text("\n".join(out),encoding="utf-8")

class Scene:
    def __init__(self):
        self.rng=np.random.default_rng(260829)
        self.stars=[(self.rng.uniform(0,W),self.rng.uniform(0,H),self.rng.uniform(.3,2.0)*S,self.rng.uniform(60,210)) for _ in range(280 if QUICK else 600)]
        self.gals=[]
        for i in range(170 if QUICK else 430):
            r=min(1.0,float(self.rng.gamma(1.4,.25)))
            th=float(self.rng.uniform(0,math.tau))
            x,y=math.cos(th)*r,math.sin(th)*r*float(self.rng.uniform(.72,1))
            size=float(np.clip(self.rng.lognormal(-.1,.55),.45,3.4))*(2.5 if i<3 else 1)
            if i<3: x*=.18; y*=.18
            self.gals.append((x,y,size,float(self.rng.uniform(.35,.82)),float(self.rng.uniform(0,1)),float(self.rng.uniform(0,math.tau))))
        self.base=self._background()
        yy,xx=np.mgrid[0:H,0:W]; nx=(xx-W/2)/(W/2); ny=(yy-H/2)/(H/2)
        self.vig=np.clip(1-.40*np.sqrt(nx*nx+ny*ny)**1.8,0,1).astype(np.float32)

    def _background(self):
        arr=np.zeros((H,W,4),dtype=np.uint8); top=np.array(C["space2"]); bot=np.array(C["space"])
        for y in range(H):
            u=y/max(H-1,1); rgb=(top*(1-u)+bot*u).astype(np.uint8); arr[y,:,:3]=rgb; arr[y,:,3]=255
        im=Image.fromarray(arr,"RGBA"); d=ImageDraw.Draw(im)
        for x,y,r,a in self.stars: d.ellipse((x-r,y-r,x+r,y+r),fill=(220,232,255,int(a)))
        return im

    def bg(self,t):
        im=self.base.copy(); haze=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(haze)
        for k,col in enumerate((C["violet"],C["cyan"],C["blue"])):
            cx=W*(.22+.30*k)+math.sin(t*(.08+.02*k))*35*S; cy=H*(.34+.11*((k+1)%2)); r=W*(.22+.04*k)
            d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=col+(10,))
        haze=haze.filter(ImageFilter.GaussianBlur(max(12,int(65*S)))); im.alpha_composite(haze); return im

    def galaxy(self,img,g,x,y,scale=1,alpha=1):
        gx,gy,sz,ell,warm,ang=g; major=max(2,sz*8*S*scale); minor=major*ell
        col=tuple(int(lerp(C["cyan"][i],C["gold"][i],warm)) for i in range(3))
        r=int(max(10,major*2.7)); tile=Image.new("RGBA",(2*r+1,2*r+1),(0,0,0,0)); d=ImageDraw.Draw(tile); c=r
        d.ellipse((c-major,c-minor,c+major,c+minor),fill=col+(int(210*alpha),))
        d.ellipse((c-minor*.25,c-minor*.25,c+minor*.25,c+minor*.25),fill=C["white"]+(int(235*alpha),))
        tile=tile.filter(ImageFilter.GaussianBlur(max(.4,.9*S))).rotate(math.degrees(ang),resample=Image.Resampling.BICUBIC)
        img.alpha_composite(tile,(int(x-r),int(y-r)))

    def cluster(self,img,t,reveal=1,scale=1,alpha=1):
        n=int(len(self.gals)*clamp(reveal)); rot=t*.012; cs,ss=math.cos(rot),math.sin(rot)
        for g in self.gals[:n]:
            gx,gy,*_=g; xr=gx*cs-gy*ss; yr=gx*ss+gy*cs; x=W*.5+xr*W*.41*scale; y=H*.47+yr*W*.41*scale
            self.galaxy(img,g,x,y,scale,alpha)

    def hud(self,img,name):
        text(img,CFG["title"],(int(44*S),int(75*S)),max(13,int(35*S)),C["white"]+(248,),True,True,stroke=1)
        label={"one":"STEP 1 // SCALE","scale":"STEP 2 // MANY GALAXIES","gravity":"STEP 3 // GRAVITY","gas":"STEP 4 // HOT GAS","dark":"STEP 5 // DARK MATTER","lens":"STEP 6 // BENT LIGHT","web":"FINAL // COSMIC WEB"}[name]
        text(img,label,(int(46*S),int(121*S)),max(8,int(18*S)),C["cyan"]+(210,),True,True,stroke=1)

    def caption(self,img,t):
        s=cap_at(t)
        if not s: return
        box=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(box); l=int(58*S); r=W-l; top=int(H*.885); bot=H-int(38*S)
        d.rounded_rectangle((l,top,r,bot),radius=max(8,int(18*S)),fill=(2,5,15,150),outline=C["grid"]+(48,),width=1); img.alpha_composite(box)
        wrapped(img,s,(l+int(24*S),top+int(14*S)),r-l-int(48*S),max(15,int(30*S)),C["white"]+(235,),max(2,int(4*S)))

    def render(self,t):
        name,a,b=shot_at(t); p=smoother((t-a)/max(b-a,1e-9)); img=self.bg(t)
        cx,cy=W*.5,H*.47
        if name=="one":
            g=self.gals[0]; self.galaxy(img,g,cx,cy,lerp(3.4,2.2,p),1)
            if p>.3:
                text(img,"ONE GALAXY",(W//2,int(H*.70)),max(12,int(39*S)),C["white"]+(245,),True,True,"ma",1)
                text(img,"BILLIONS OF STARS",(W//2,int(H*.75)),max(8,int(20*S)),C["muted"]+(220,),True,True,"ma",1)
        elif name=="scale":
            self.cluster(img,t,p,lerp(1.2,.94,p),.98)
            val=int(lerp(1,1000,smooth(max(0,(p-.15)/.75))))
            text(img,f"{val:,}+",(W//2,int(H*.70)),max(25,int(100*S)),C["white"]+(255,),True,True,"ma",2)
            text(img,"GALAXIES CAN SHARE ONE CLUSTER",(W//2,int(H*.79)),max(9,int(28*S)),C["cyan"]+(235,),True,True,"ma",1)
        elif name=="gravity":
            self.cluster(img,t,1,.91,.82); ov=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(ov)
            for i in range(7):
                u=(i+(t*.25)%1)/7; rx=W*(.08+u*.34); ry=rx*.56; d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),outline=C["blue"]+(int(75*(1-u)),),width=max(1,int(2*S)))
            img.alpha_composite(ov)
            text(img,"GRAVITY BINDS THE SYSTEM",(W//2,int(H*.75)),max(10,int(32*S)),C["white"]+(245,),True,True,"ma",1)
            text(img,"GALAXIES ORBIT A SHARED MASS",(W//2,int(H*.80)),max(8,int(21*S)),C["muted"]+(225,),True,True,"ma",1)
        elif name=="gas":
            self.cluster(img,t,1,.92,.60); gas=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(gas)
            for i in range(8):
                th=i*math.tau/8+t*.03; rr=W*(.14+.014*i); x=cx+math.cos(th)*W*.18; y=cy+math.sin(th)*W*.11
                d.ellipse((x-rr,y-rr*.65,x+rr,y+rr*.65),fill=C["gas"]+(int(20+8*p),))
            gas=gas.filter(ImageFilter.GaussianBlur(max(15,int(70*S)))); img.alpha_composite(gas)
            text(img,"HOT INTRACLUSTER GAS",(W//2,int(H*.71)),max(10,int(34*S)),C["cyan"]+(250,),True,True,"ma",1)
            text(img,"TENS OF MILLIONS OF KELVIN",(W//2,int(H*.77)),max(9,int(24*S)),C["white"]+(240,),True,True,"ma",1)
            text(img,"VISIBLE IN X-RAYS",(W//2,int(H*.82)),max(8,int(19*S)),C["muted"]+(220,),True,True,"ma",1)
        elif name=="dark":
            self.cluster(img,t,1,.92,.70); halo=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(halo)
            for i in range(7):
                rx=W*(.10+i*.045); ry=rx*.62; d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),outline=C["violet"]+(int((125-i*13)*p),),width=max(1,int(2*S)))
            halo=halo.filter(ImageFilter.GaussianBlur(max(1,int(2*S)))); img.alpha_composite(halo)
            text(img,"MOST OF THE MASS IS INVISIBLE",(W//2,int(H*.72)),max(9,int(31*S)),C["white"]+(250,),True,True,"ma",1)
            text(img,"DARK MATTER",(W//2,int(H*.79)),max(18,int(66*S)),C["violet"]+(255,),True,True,"ma",2)
            text(img,"MAPPED BY GRAVITY — NOT BY LIGHT",(W//2,int(H*.85)),max(7,int(18*S)),C["muted"]+(220,),True,True,"ma",1)
        elif name=="lens":
            self.cluster(img,t,1,.88,.72); ov=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(ov)
            for st,en,rad,wid in [(198,278,.28,4),(18,82,.31,3),(102,148,.245,3),(306,342,.22,2)]:
                r=W*rad; box=(cx-r,cy-r*.78,cx+r,cy+r*.78); d.arc(box,st,en,fill=C["gold"]+(int(230*p),),width=max(1,int(wid*S)))
            img.alpha_composite(ov)
            text(img,"THE CLUSTER BENDS LIGHT",(W//2,int(H*.72)),max(10,int(34*S)),C["white"]+(250,),True,True,"ma",1)
            text(img,"GRAVITATIONAL LENSING",(W//2,int(H*.79)),max(12,int(42*S)),C["gold"]+(255,),True,True,"ma",2)
        else:
            self.cluster(img,t,1,lerp(.80,.34,p),.50); web=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(web)
            pts=[(W*.10,H*.32),(W*.28,H*.58),(W*.50,H*.47),(W*.68,H*.30),(W*.84,H*.60),(W*.90,H*.40),(W*.55,H*.18),(W*.18,H*.18)]
            edges=[(0,1),(1,2),(2,3),(3,5),(2,4),(4,5),(1,6),(6,3),(1,7),(7,6),(2,6)]
            for u,v in edges: d.line((*pts[u],*pts[v]),fill=C["cyan"]+(int(62*p),),width=max(1,int(2*S)))
            for x,y in pts: r=6*S; d.ellipse((x-r,y-r,x+r,y+r),fill=C["white"]+(int(190*p),))
            web=web.filter(ImageFilter.GaussianBlur(max(.4,1.2*S))); img.alpha_composite(web)
            text(img,"A NODE IN THE COSMIC WEB",(W//2,int(H*.73)),max(10,int(34*S)),C["cyan"]+(245,),True,True,"ma",1)
            text(img,"GALAXY CLUSTER",(W//2,int(H*.81)),max(18,int(64*S)),C["white"]+(255,),True,True,"ma",2)

        self.hud(img,name); self.caption(img,t)
        text(img,"SCHEMATIC VISUALIZATION // NASA / ESA SCIENCE FRAMING",(int(44*S),H-int(18*S)),max(6,int(12*S)),C["muted"]+(145,),True,True,"ls",1)
        arr=np.asarray(img.convert("RGB")).astype(np.float32); arr*=self.vig[...,None]
        rng=np.random.default_rng(int(t*1000)+90210); arr=np.clip(arr+rng.normal(0,3.5,arr.shape[:2])[...,None],0,255).astype(np.uint8)
        return np.asarray(ImageEnhance.Contrast(Image.fromarray(arr)).enhance(1.08))

def soundtrack(path):
    sr=int(CFG["sr"]); n=int(round(sr*DUR)); t=np.arange(n)/sr; rng=np.random.default_rng(5511)
    a=.08*np.sin(math.tau*29*t+.4*np.sin(math.tau*.05*t))+.045*np.sin(math.tau*43*t+.9)+.022*np.sin(math.tau*71*t)
    controls=rng.normal(0,1,max(8,int(DUR*5))); a += .02*np.interp(t,np.linspace(0,DUR,len(controls)),controls)
    for _,st,_ in SHOTS[1:]:
        env=np.exp(-.5*((t-st)/(.12 if QUICK else .24))**2); a += .035*env*np.sin(math.tau*(80+30*t/DUR)*t)
    intro=np.clip(t/max(1.2,DUR*.06),0,1); intro=intro*intro*(3-2*intro); ox=np.clip((t-(DUR-1.1))/1.0,0,1); out=1-ox*ox*(3-2*ox); a*=intro*out
    a=np.clip(a/max(np.max(np.abs(a)),1e-9)*.86,-1,1); pcm=(a*32767).astype(np.int16)
    with wave.open(str(path),"wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(pcm.tobytes())

def ffmpeg_exe():
    if imageio_ffmpeg:
        try: return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception: pass
    return shutil.which("ffmpeg")

def mux(video,audio,out):
    ff=ffmpeg_exe()
    if not ff: return False
    try:
        subprocess.run([ff,"-y","-i",str(video),"-i",str(audio),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return out.exists() and out.stat().st_size>0
    except Exception: return False

def notes():
    p=DATA/"science_notes.json"
    p.write_text(json.dumps({
        "title":CFG["title"],"visual_status":"schematic_procedural_visualization",
        "science_points":[
            "Rich galaxy clusters contain hundreds to thousands of galaxies.",
            "Intracluster gas can reach tens of millions of kelvin and emits X-rays.",
            "Dark matter supplies most of the gravitating mass in galaxy clusters.",
            "Massive clusters gravitationally lens background galaxies.",
            "Galaxy clusters occupy dense nodes of the cosmic web."],
        "references":[
            "https://science.nasa.gov/universe/galaxies/large-scale-structures/",
            "https://science.nasa.gov/dark-matter/",
            "https://www.esa.int/ESA_Multimedia/Images/2015/12/Galaxy_Cluster_Abell_2744"]},indent=2),encoding="utf-8")
    return p


# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'What Is a Galaxy Cluster? 🌌 #rootjatin'
YOUTUBE_DESCRIPTION = 'A galaxy cluster is a vast gravitationally bound structure containing hundreds or thousands of galaxies along with hot intracluster gas and much more mass in dark matter. This educational visualization explains the different components and how astronomers infer the unseen mass through its gravitational effects on motion and light. The scene is schematic, not a map of one specific observed cluster.'
YOUTUBE_HASHTAGS = '#rootjatin #GalaxyCluster #Galaxies #DarkMatter #Astronomy #Space #Cosmology #Science'

def write_youtube_metadata_txt() -> Path:
    cfg = globals().get("CONFIG") or globals().get("CFG") or {}
    basename = (
        cfg.get("basename")
        or cfg.get("output_basename")
        or cfg.get("file_stem")
        or Path(__file__).stem
    )
    root = globals().get("OUTPUT_ROOT") or globals().get("ROOT") or Path(".")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{basename}_title_description.txt"
    path.write_text(
        "TITLE\n" + YOUTUBE_TITLE +
        "\n\nDESCRIPTION\n" + YOUTUBE_DESCRIPTION +
        "\n\nHASHTAGS\n" + YOUTUBE_HASHTAGS + "\n",
        encoding="utf-8",
    )
    return path

def main():
    print("Title:",CFG["title"]); print("Science notes:",notes().resolve()); sc=Scene()
    for pt in [1.1,DUR*.22,DUR*.40,DUR*.56,DUR*.72,DUR*.86,DUR-.5]:
        Image.fromarray(sc.render(float(pt))).save(PREV/f"preview_{int(pt):02d}s.png")
    srt=ROOT/f"{CFG['basename']}_subtitles.srt"; write_srt(srt)
    raw=ROOT/f"{CFG['basename']}_silent.mp4"; final=ROOT/f"{CFG['basename']}_final.mp4"; audio=ROOT/f"{CFG['basename']}_ambient.wav"
    times=np.arange(int(round(DUR*FPS)))/FPS
    with iio.get_writer(raw,fps=FPS,codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as wr:
        for t in tqdm(times,desc="Rendering galaxy-cluster short"): wr.append_data(sc.render(float(t)))
    soundtrack(audio)
    if not mux(raw,audio,final): shutil.copyfile(raw,final)
    print("Final:",final.resolve())
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())

from __future__ import annotations

"""Cinematic vertical YouTube Short: WHAT IS A GALAXY CLUSTER?

Schematic science visualization: one galaxy -> cluster scale -> gravity -> hot gas
-> dark matter -> gravitational lensing -> cosmic web.

Install:
    pip install numpy pillow imageio imageio-ffmpeg tqdm
Quick:
    GALAXY_CLUSTER_SHORT_QUICK=1 python what_is_a_galaxy_cluster.py
"""

import json, math, os, shutil, subprocess, wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm.auto import tqdm
try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

QUICK = os.environ.get("GALAXY_CLUSTER_SHORT_QUICK", "0") == "1"
ROOT = Path("what_is_a_galaxy_cluster_output")
PREV = ROOT / "previews"
DATA = ROOT / "data"
for p in (ROOT, PREV, DATA): p.mkdir(parents=True, exist_ok=True)

CFG: Dict[str, Any] = {
    "w": 540 if QUICK else 1080,
    "h": 960 if QUICK else 1920,
    "fps": 6 if QUICK else 24,
    "duration": 12.0 if QUICK else 58.0,
    "title": "WHAT IS A GALAXY CLUSTER?",
    "basename": "what_is_a_galaxy_cluster",
    "sr": 22050 if QUICK else 44100,
}
W, H, FPS, DUR = int(CFG["w"]), int(CFG["h"]), int(CFG["fps"]), float(CFG["duration"])
S = W / 1080.0
C = {
    "space": (2,4,13), "space2": (5,8,22), "white": (247,250,255),
    "muted": (165,187,215), "cyan": (87,221,255), "blue": (92,145,255),
    "violet": (181,111,255), "gold": (255,205,112), "gas": (119,191,255),
    "grid": (94,127,176),
}

FULL_SHOTS = [
    ("one",0,7), ("scale",7,17), ("gravity",17,27), ("gas",27,37),
    ("dark",37,46), ("lens",46,53), ("web",53,58),
]
FULL_CAPS = [
    (0.5,6.6,"A galaxy is already enormous. A galaxy cluster is hundreds to thousands of galaxies gathered into one gravitational system."),
    (7.3,16.6,"The galaxies can span millions of light-years, moving through a shared gravitational potential."),
    (17.3,26.6,"Gravity binds the cluster together. The visible galaxies are only part of what is there."),
    (27.3,36.6,"Between the galaxies is extremely hot, thin gas. It shines strongly in X-rays and can reach tens of millions of degrees."),
    (37.3,45.6,"Most of the cluster's gravitating mass is invisible dark matter, mapped through its effects on motion and light."),
    (46.3,52.6,"A massive cluster bends spacetime enough to stretch and magnify more distant galaxies behind it."),
    (53.1,57.7,"Galaxy clusters sit at dense intersections of the cosmic web: some of the largest structures held together by gravity."),
]
scale_t = DUR/58.0 if QUICK else 1.0
SHOTS = [(n,a*scale_t,b*scale_t) for n,a,b in FULL_SHOTS]
CAPS = [(a*scale_t,b*scale_t,t) for a,b,t in FULL_CAPS]

def clamp(x,a=0.0,b=1.0): return max(a,min(b,float(x)))
def smooth(x):
    x=clamp(x); return x*x*(3-2*x)
def smoother(x):
    x=clamp(x); return x*x*x*(x*(x*6-15)+10)
def lerp(a,b,t): return a+(b-a)*t

def shot_at(t):
    for n,a,b in SHOTS:
        if a <= t < b: return n,a,b
    return SHOTS[-1]

def cap_at(t):
    """Return a short readable page of the active caption."""
    for start, end, value in CAPS:
        if start <= t < end:
            words = str(value).split()
            if not words:
                return None
            target_words = 18
            page_count = max(1, (len(words) + target_words - 1) // target_words)
            base = len(words) // page_count
            extra = len(words) % page_count
            progress = (t - start) / max(end - start, 1e-9)
            page = min(page_count - 1, max(0, int(progress * page_count)))
            first = page * base + min(page, extra)
            count = base + (1 if page < extra else 0)
            return " ".join(words[first:first + count])
    return None

def font(size,bold=False,condensed=False):
    candidates=[]
    if condensed and bold: candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf","DejaVuSansCondensed-Bold.ttf"]
    if condensed: candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf","DejaVuSansCondensed.ttf"]
    candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for f in candidates:
        try: return ImageFont.truetype(f,size=size)
        except Exception: pass
    return ImageFont.load_default()

def text(img,s,xy,size,fill,bold=False,condensed=False,anchor="la",stroke=2):
    ImageDraw.Draw(img).text(xy,s,font=font(size,bold,condensed),fill=fill,anchor=anchor,stroke_width=stroke,stroke_fill=(0,0,0,min(230,fill[3])))

def wrapped(img,s,xy,maxw,size,fill,spacing=4):
    d=ImageDraw.Draw(img); f=font(size); lines=[]; cur=""
    for word in s.split():
        q=word if not cur else cur+" "+word
        if d.textbbox((0,0),q,font=f,stroke_width=1)[2] <= maxw: cur=q
        else:
            if cur: lines.append(cur)
            cur=word
    if cur: lines.append(cur)
    x,y=xy
    for line in lines[:4]:
        d.text((x,y),line,font=f,fill=fill,stroke_width=2,stroke_fill=(0,0,0,220))
        y += d.textbbox((x,y),line,font=f,stroke_width=2)[3]-y + spacing

def srt_time(sec):
    ms=int(round(sec*1000)); h=ms//3600000; ms%=3600000; m=ms//60000; ms%=60000; s=ms//1000; ms%=1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(path):
    out=[]
    for i,(a,b,s) in enumerate(CAPS,1): out += [str(i),f"{srt_time(a)} --> {srt_time(b)}",s,""]
    path.write_text("\n".join(out),encoding="utf-8")

class Scene:
    def __init__(self):
        self.rng=np.random.default_rng(260829)
        self.stars=[(self.rng.uniform(0,W),self.rng.uniform(0,H),self.rng.uniform(.3,2.0)*S,self.rng.uniform(60,210)) for _ in range(280 if QUICK else 600)]
        self.gals=[]
        for i in range(170 if QUICK else 430):
            r=min(1.0,float(self.rng.gamma(1.4,.25)))
            th=float(self.rng.uniform(0,math.tau))
            x,y=math.cos(th)*r,math.sin(th)*r*float(self.rng.uniform(.72,1))
            size=float(np.clip(self.rng.lognormal(-.1,.55),.45,3.4))*(2.5 if i<3 else 1)
            if i<3: x*=.18; y*=.18
            self.gals.append((x,y,size,float(self.rng.uniform(.35,.82)),float(self.rng.uniform(0,1)),float(self.rng.uniform(0,math.tau))))
        self.base=self._background()
        yy,xx=np.mgrid[0:H,0:W]; nx=(xx-W/2)/(W/2); ny=(yy-H/2)/(H/2)
        self.vig=np.clip(1-.40*np.sqrt(nx*nx+ny*ny)**1.8,0,1).astype(np.float32)

    def _background(self):
        arr=np.zeros((H,W,4),dtype=np.uint8); top=np.array(C["space2"]); bot=np.array(C["space"])
        for y in range(H):
            u=y/max(H-1,1); rgb=(top*(1-u)+bot*u).astype(np.uint8); arr[y,:,:3]=rgb; arr[y,:,3]=255
        im=Image.fromarray(arr,"RGBA"); d=ImageDraw.Draw(im)
        for x,y,r,a in self.stars: d.ellipse((x-r,y-r,x+r,y+r),fill=(220,232,255,int(a)))
        return im

    def bg(self,t):
        im=self.base.copy(); haze=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(haze)
        for k,col in enumerate((C["violet"],C["cyan"],C["blue"])):
            cx=W*(.22+.30*k)+math.sin(t*(.08+.02*k))*35*S; cy=H*(.34+.11*((k+1)%2)); r=W*(.22+.04*k)
            d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=col+(10,))
        haze=haze.filter(ImageFilter.GaussianBlur(max(12,int(65*S)))); im.alpha_composite(haze); return im

    def galaxy(self,img,g,x,y,scale=1,alpha=1):
        gx,gy,sz,ell,warm,ang=g; major=max(2,sz*8*S*scale); minor=major*ell
        col=tuple(int(lerp(C["cyan"][i],C["gold"][i],warm)) for i in range(3))
        r=int(max(10,major*2.7)); tile=Image.new("RGBA",(2*r+1,2*r+1),(0,0,0,0)); d=ImageDraw.Draw(tile); c=r
        d.ellipse((c-major,c-minor,c+major,c+minor),fill=col+(int(210*alpha),))
        d.ellipse((c-minor*.25,c-minor*.25,c+minor*.25,c+minor*.25),fill=C["white"]+(int(235*alpha),))
        tile=tile.filter(ImageFilter.GaussianBlur(max(.4,.9*S))).rotate(math.degrees(ang),resample=Image.Resampling.BICUBIC)
        img.alpha_composite(tile,(int(x-r),int(y-r)))

    def cluster(self,img,t,reveal=1,scale=1,alpha=1):
        n=int(len(self.gals)*clamp(reveal)); rot=t*.012; cs,ss=math.cos(rot),math.sin(rot)
        for g in self.gals[:n]:
            gx,gy,*_=g; xr=gx*cs-gy*ss; yr=gx*ss+gy*cs; x=W*.5+xr*W*.41*scale; y=H*.47+yr*W*.41*scale
            self.galaxy(img,g,x,y,scale,alpha)

    def hud(self,img,name):
        text(img,CFG["title"],(int(44*S),int(75*S)),max(13,int(35*S)),C["white"]+(248,),True,True,stroke=1)
        label={"one":"STEP 1 // SCALE","scale":"STEP 2 // MANY GALAXIES","gravity":"STEP 3 // GRAVITY","gas":"STEP 4 // HOT GAS","dark":"STEP 5 // DARK MATTER","lens":"STEP 6 // BENT LIGHT","web":"FINAL // COSMIC WEB"}[name]
        text(img,label,(int(46*S),int(121*S)),max(8,int(18*S)),C["cyan"]+(210,),True,True,stroke=1)

    def caption(self,img,t):
        s=cap_at(t)
        if not s: return
        box=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(box); l=int(58*S); r=W-l; top=int(H*.885); bot=H-int(38*S)
        d.rounded_rectangle((l,top,r,bot),radius=max(8,int(18*S)),fill=(2,5,15,150),outline=C["grid"]+(48,),width=1); img.alpha_composite(box)
        wrapped(img,s,(l+int(24*S),top+int(14*S)),r-l-int(48*S),max(15,int(30*S)),C["white"]+(235,),max(2,int(4*S)))

    def render(self,t):
        name,a,b=shot_at(t); p=smoother((t-a)/max(b-a,1e-9)); img=self.bg(t)
        cx,cy=W*.5,H*.47
        if name=="one":
            g=self.gals[0]; self.galaxy(img,g,cx,cy,lerp(3.4,2.2,p),1)
            if p>.3:
                text(img,"ONE GALAXY",(W//2,int(H*.70)),max(12,int(39*S)),C["white"]+(245,),True,True,"ma",1)
                text(img,"BILLIONS OF STARS",(W//2,int(H*.75)),max(8,int(20*S)),C["muted"]+(220,),True,True,"ma",1)
        elif name=="scale":
            self.cluster(img,t,p,lerp(1.2,.94,p),.98)
            val=int(lerp(1,1000,smooth(max(0,(p-.15)/.75))))
            text(img,f"{val:,}+",(W//2,int(H*.70)),max(25,int(100*S)),C["white"]+(255,),True,True,"ma",2)
            text(img,"GALAXIES CAN SHARE ONE CLUSTER",(W//2,int(H*.79)),max(9,int(28*S)),C["cyan"]+(235,),True,True,"ma",1)
        elif name=="gravity":
            self.cluster(img,t,1,.91,.82); ov=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(ov)
            for i in range(7):
                u=(i+(t*.25)%1)/7; rx=W*(.08+u*.34); ry=rx*.56; d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),outline=C["blue"]+(int(75*(1-u)),),width=max(1,int(2*S)))
            img.alpha_composite(ov)
            text(img,"GRAVITY BINDS THE SYSTEM",(W//2,int(H*.75)),max(10,int(32*S)),C["white"]+(245,),True,True,"ma",1)
            text(img,"GALAXIES ORBIT A SHARED MASS",(W//2,int(H*.80)),max(8,int(21*S)),C["muted"]+(225,),True,True,"ma",1)
        elif name=="gas":
            self.cluster(img,t,1,.92,.60); gas=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(gas)
            for i in range(8):
                th=i*math.tau/8+t*.03; rr=W*(.14+.014*i); x=cx+math.cos(th)*W*.18; y=cy+math.sin(th)*W*.11
                d.ellipse((x-rr,y-rr*.65,x+rr,y+rr*.65),fill=C["gas"]+(int(20+8*p),))
            gas=gas.filter(ImageFilter.GaussianBlur(max(15,int(70*S)))); img.alpha_composite(gas)
            text(img,"HOT INTRACLUSTER GAS",(W//2,int(H*.71)),max(10,int(34*S)),C["cyan"]+(250,),True,True,"ma",1)
            text(img,"TENS OF MILLIONS OF KELVIN",(W//2,int(H*.77)),max(9,int(24*S)),C["white"]+(240,),True,True,"ma",1)
            text(img,"VISIBLE IN X-RAYS",(W//2,int(H*.82)),max(8,int(19*S)),C["muted"]+(220,),True,True,"ma",1)
        elif name=="dark":
            self.cluster(img,t,1,.92,.70); halo=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(halo)
            for i in range(7):
                rx=W*(.10+i*.045); ry=rx*.62; d.ellipse((cx-rx,cy-ry,cx+rx,cy+ry),outline=C["violet"]+(int((125-i*13)*p),),width=max(1,int(2*S)))
            halo=halo.filter(ImageFilter.GaussianBlur(max(1,int(2*S)))); img.alpha_composite(halo)
            text(img,"MOST OF THE MASS IS INVISIBLE",(W//2,int(H*.72)),max(9,int(31*S)),C["white"]+(250,),True,True,"ma",1)
            text(img,"DARK MATTER",(W//2,int(H*.79)),max(18,int(66*S)),C["violet"]+(255,),True,True,"ma",2)
            text(img,"MAPPED BY GRAVITY — NOT BY LIGHT",(W//2,int(H*.85)),max(7,int(18*S)),C["muted"]+(220,),True,True,"ma",1)
        elif name=="lens":
            self.cluster(img,t,1,.88,.72); ov=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(ov)
            for st,en,rad,wid in [(198,278,.28,4),(18,82,.31,3),(102,148,.245,3),(306,342,.22,2)]:
                r=W*rad; box=(cx-r,cy-r*.78,cx+r,cy+r*.78); d.arc(box,st,en,fill=C["gold"]+(int(230*p),),width=max(1,int(wid*S)))
            img.alpha_composite(ov)
            text(img,"THE CLUSTER BENDS LIGHT",(W//2,int(H*.72)),max(10,int(34*S)),C["white"]+(250,),True,True,"ma",1)
            text(img,"GRAVITATIONAL LENSING",(W//2,int(H*.79)),max(12,int(42*S)),C["gold"]+(255,),True,True,"ma",2)
        else:
            self.cluster(img,t,1,lerp(.80,.34,p),.50); web=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(web)
            pts=[(W*.10,H*.32),(W*.28,H*.58),(W*.50,H*.47),(W*.68,H*.30),(W*.84,H*.60),(W*.90,H*.40),(W*.55,H*.18),(W*.18,H*.18)]
            edges=[(0,1),(1,2),(2,3),(3,5),(2,4),(4,5),(1,6),(6,3),(1,7),(7,6),(2,6)]
            for u,v in edges: d.line((*pts[u],*pts[v]),fill=C["cyan"]+(int(62*p),),width=max(1,int(2*S)))
            for x,y in pts: r=6*S; d.ellipse((x-r,y-r,x+r,y+r),fill=C["white"]+(int(190*p),))
            web=web.filter(ImageFilter.GaussianBlur(max(.4,1.2*S))); img.alpha_composite(web)
            text(img,"A NODE IN THE COSMIC WEB",(W//2,int(H*.73)),max(10,int(34*S)),C["cyan"]+(245,),True,True,"ma",1)
            text(img,"GALAXY CLUSTER",(W//2,int(H*.81)),max(18,int(64*S)),C["white"]+(255,),True,True,"ma",2)

        self.hud(img,name); self.caption(img,t)
        text(img,"SCHEMATIC VISUALIZATION // NASA / ESA SCIENCE FRAMING",(int(44*S),H-int(18*S)),max(6,int(12*S)),C["muted"]+(145,),True,True,"ls",1)
        arr=np.asarray(img.convert("RGB")).astype(np.float32); arr*=self.vig[...,None]
        rng=np.random.default_rng(int(t*1000)+90210); arr=np.clip(arr+rng.normal(0,3.5,arr.shape[:2])[...,None],0,255).astype(np.uint8)
        return np.asarray(ImageEnhance.Contrast(Image.fromarray(arr)).enhance(1.08))

def soundtrack(path):
    sr=int(CFG["sr"]); n=int(round(sr*DUR)); t=np.arange(n)/sr; rng=np.random.default_rng(5511)
    a=.08*np.sin(math.tau*29*t+.4*np.sin(math.tau*.05*t))+.045*np.sin(math.tau*43*t+.9)+.022*np.sin(math.tau*71*t)
    controls=rng.normal(0,1,max(8,int(DUR*5))); a += .02*np.interp(t,np.linspace(0,DUR,len(controls)),controls)
    for _,st,_ in SHOTS[1:]:
        env=np.exp(-.5*((t-st)/(.12 if QUICK else .24))**2); a += .035*env*np.sin(math.tau*(80+30*t/DUR)*t)
    intro=np.clip(t/max(1.2,DUR*.06),0,1); intro=intro*intro*(3-2*intro); ox=np.clip((t-(DUR-1.1))/1.0,0,1); out=1-ox*ox*(3-2*ox); a*=intro*out
    a=np.clip(a/max(np.max(np.abs(a)),1e-9)*.86,-1,1); pcm=(a*32767).astype(np.int16)
    with wave.open(str(path),"wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(pcm.tobytes())

def ffmpeg_exe():
    if imageio_ffmpeg:
        try: return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception: pass
    return shutil.which("ffmpeg")

def mux(video,audio,out):
    ff=ffmpeg_exe()
    if not ff: return False
    try:
        subprocess.run([ff,"-y","-i",str(video),"-i",str(audio),"-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(out)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return out.exists() and out.stat().st_size>0
    except Exception: return False

def notes():
    p=DATA/"science_notes.json"
    p.write_text(json.dumps({
        "title":CFG["title"],"visual_status":"schematic_procedural_visualization",
        "science_points":[
            "Rich galaxy clusters contain hundreds to thousands of galaxies.",
            "Intracluster gas can reach tens of millions of kelvin and emits X-rays.",
            "Dark matter supplies most of the gravitating mass in galaxy clusters.",
            "Massive clusters gravitationally lens background galaxies.",
            "Galaxy clusters occupy dense nodes of the cosmic web."],
        "references":[
            "https://science.nasa.gov/universe/galaxies/large-scale-structures/",
            "https://science.nasa.gov/dark-matter/",
            "https://www.esa.int/ESA_Multimedia/Images/2015/12/Galaxy_Cluster_Abell_2744"]},indent=2),encoding="utf-8")
    return p


# =============================================================================
# YouTube title / description sidecar
# =============================================================================

YOUTUBE_TITLE = 'What Is a Galaxy Cluster? 🌌 #rootjatin'
YOUTUBE_DESCRIPTION = 'A galaxy cluster is a vast gravitationally bound structure containing hundreds or thousands of galaxies along with hot intracluster gas and much more mass in dark matter. This educational visualization explains the different components and how astronomers infer the unseen mass through its gravitational effects on motion and light. The scene is schematic, not a map of one specific observed cluster.'
YOUTUBE_HASHTAGS = '#rootjatin #GalaxyCluster #Galaxies #DarkMatter #Astronomy #Space #Cosmology #Science'

def write_youtube_metadata_txt() -> Path:
    cfg = globals().get("CONFIG") or globals().get("CFG") or {}
    basename = (
        cfg.get("basename")
        or cfg.get("output_basename")
        or cfg.get("file_stem")
        or Path(__file__).stem
    )
    root = globals().get("OUTPUT_ROOT") or globals().get("ROOT") or Path(".")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{basename}_title_description.txt"
    path.write_text(
        "TITLE\n" + YOUTUBE_TITLE +
        "\n\nDESCRIPTION\n" + YOUTUBE_DESCRIPTION +
        "\n\nHASHTAGS\n" + YOUTUBE_HASHTAGS + "\n",
        encoding="utf-8",
    )
    return path

def main():
    print("Title:",CFG["title"]); print("Science notes:",notes().resolve()); sc=Scene()
    for pt in [1.1,DUR*.22,DUR*.40,DUR*.56,DUR*.72,DUR*.86,DUR-.5]:
        Image.fromarray(sc.render(float(pt))).save(PREV/f"preview_{int(pt):02d}s.png")
    srt=ROOT/f"{CFG['basename']}_subtitles.srt"; write_srt(srt)
    raw=ROOT/f"{CFG['basename']}_silent.mp4"; final=ROOT/f"{CFG['basename']}_final.mp4"; audio=ROOT/f"{CFG['basename']}_ambient.wav"
    times=np.arange(int(round(DUR*FPS)))/FPS
    with iio.get_writer(raw,fps=FPS,codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=None) as wr:
        for t in tqdm(times,desc="Rendering galaxy-cluster short"): wr.append_data(sc.render(float(t)))
    soundtrack(audio)
    if not mux(raw,audio,final): shutil.copyfile(raw,final)
    print("Final:",final.resolve())
    metadata_txt = write_youtube_metadata_txt()
    print("Title/description TXT:", metadata_txt.resolve())

if __name__=="__main__": main()
