from cinematic_space_engine_v2 import *


def draw(r, img, t, sh, p):
    c=(r.W*0.5, r.H*0.44)
    if sh.name == 'reveal':
        r.planet(img, c, r.W*0.11, base=(38,112,190), rim=(120,220,255))
        for i in range(100):
            a = i/100*math.tau + t*0.2*(1+0.3*(i%5))
            rr = r.W*(0.16 + 0.23*(i/100))
            pt = (c[0]+math.cos(a)*rr, c[1]+math.sin(a)*rr*(0.33+0.09*(i%4)))
            r.glow_circle(img, pt, r.W*0.0045, (245,232,214), core=(255,245,228,255), blur=5)
        r.text_center(img, '100 MOONS AROUND ONE PLANET', int(r.H*0.18), size=52)
    elif sh.name == 'shells':
        r.planet(img, c, r.W*0.11, base=(35,104,188), rim=(120,220,255))
        for k in range(5):
            rr = r.W*(0.18+0.06*k)
            r.orbit(img, c, rr, rr*(0.32+0.03*k), alpha=90 if k<3 else 55, width=2, dash=(k%2==1))
        for i in range(100):
            lane = i % 5
            a = i/100*math.tau + lane*0.3 + t*0.18*(1+lane*0.2)
            rr = r.W*(0.18+0.06*lane)
            pt = (c[0]+math.cos(a)*rr, c[1]+math.sin(a)*rr* (0.32+0.03*lane))
            size = r.W*(0.003 + 0.003*((i*7)%9)/8)
            col = (255,225,180) if lane<2 else ((100,230,255) if lane==2 else (210,180,255))
            r.glow_circle(img, pt, size, col, core=(*col,255), blur=4)
        r.text_center(img, 'INNER LANES GO FAST. OUTER ONES DRIFT.', int(r.H*0.18), size=42)
    elif sh.name == 'resonances':
        r.planet(img, c, r.W*0.11, base=(35,104,188))
        r.draw_grid(img, r.W*0.10, r.H*0.60, r.W*0.80, r.H*0.18, rows=3, cols=6)
        pts=[]
        for i in range(180):
            q=i/179
            x=r.W*0.12+r.W*0.76*q
            y=r.H*0.69-r.H*0.06*math.sin(q*math.tau*3+t*0.4)*(0.6+0.4*math.sin(q*math.tau))
            pts.append((x,y))
        r.trail(img, pts, (95,230,255), 4)
        for i in range(0,180,30):
            r.small_label(img, f'{(i//30)+1}:1', int(pts[i][0]-14*r.S), int(pts[i][1]-36*r.S), color=(255,190,110))
        r.text_center(img, 'RESONANCES START TO ORGANIZE THE CHAOS', int(r.H*0.18), size=44)
    elif sh.name == 'collisions':
        r.planet(img, c, r.W*0.11, base=(35,104,188))
        for i in range(44):
            a = i/44*math.tau + t*0.8
            rr = r.W*(0.18+0.18*(i%7)/6)
            pt = (c[0]+math.cos(a)*rr, c[1]+math.sin(a)*rr*(0.33+0.04*(i%3)))
            r.glow_circle(img, pt, r.W*0.006, (255,220,180), core=(255,238,210,255), blur=4)
        for j in range(4):
            x = c[0] + math.sin(t*0.7+j)*r.W*0.18
            y = c[1] + math.cos(t*0.9+j*0.7)*r.H*0.08
            rr = r.W*(0.010+0.004*j)
            r.glow_circle(img, (x,y), rr, (255,140,90), core=(255,230,210,255), blur=14)
        r.meter(img, int(r.W*0.16), int(r.H*0.20), int(r.W*0.68), 0.68+0.18*math.sin(t*0.6), 'INSTABILITY', color=(255,130,110))
    elif sh.name == 'survivors':
        r.planet(img, c, r.W*0.11, base=(35,104,188))
        kept=[]
        for i in range(14):
            a = i/14*math.tau + t*0.24*(1+0.1*i)
            rr = r.W*(0.20+0.23*i/13)
            pt = (c[0]+math.cos(a)*rr, c[1]+math.sin(a)*rr*(0.30+0.08*(i%4)/3))
            kept.append(pt)
            r.glow_circle(img, pt, r.W*0.008, (105,235,255), core=(220,250,255,255), blur=7)
        r.text_center(img, 'MOST SYSTEMS WOULD PRUNE THEMSELVES', int(r.H*0.18), size=43)
        r.hud(img, 'SURVIVORS', 'only a subset stays spaced out', int(r.H*0.23))
    elif sh.name == 'eclipses':
        r.planet(img, c, r.W*0.11, base=(35,104,188))
        for i in range(30):
            a = i/30*math.tau + t*0.25
            rr = r.W*(0.20+0.18*(i%6)/5)
            pt = (c[0]+math.cos(a)*rr, c[1]+math.sin(a)*rr*(0.33+0.03*(i%4)))
            r.glow_circle(img, pt, r.W*0.005, (240,235,220), core=(255,245,230,255), blur=5)
        d=ImageDraw.Draw(img,'RGBA')
        for k in range(5):
            x = r.W*(0.18+0.15*k)
            d.rectangle((x, r.H*0.26, x+r.W*0.02, r.H*0.36), fill=(90,210,255, 130 if k%2==0 else 70))
        r.text_center(img, 'THE SKY WOULD BE CONSTANTLY CROWDED', int(r.H*0.18), size=43)
    else:
        r.text_center(img, '100 MOONS IS POSSIBLE TO DRAW. HARDER TO KEEP.', int(r.H*0.18), size=42)

shots = [
    Shot('reveal', 0, 7, 'I started with a giant planet and gave it a ridiculous retinue: one hundred moons.'),
    Shot('shells', 7, 16, 'To fit them in, the moons have to spread into orbital shells. Inner moons race around much faster than outer ones.'),
    Shot('resonances', 16, 25, 'Very quickly, resonances begin to matter. Some orbits reinforce one another while others become dangerous.'),
    Shot('collisions', 25, 34, 'If too many moons crowd the same orbital zone, close encounters and collisions become almost unavoidable.'),
    Shot('survivors', 34, 43, 'Over time, the system tends to prune itself. The survivors are the moons that keep enough distance or fall into stable resonant packs.'),
    Shot('eclipses', 43, 51, 'From the planet below, the sky would be a festival of eclipses, moonrises, and overlapping orbital tracks.'),
    Shot('outro', 51, 58, 'So a planet with 100 moons is a spectacular idea — but the long-term version probably ends up with far fewer stable survivors.')
]

spec = Spec(
    title='I SIMULATED A PLANET WITH 100 MOONS',
    subtitle='orbital shells // resonances // instability pruning',
    basename='i_simulated_a_planet_with_100_moons',
    shots=shots,
    draw=draw,
    notes=[
        'This is a cinematic toy simulation concept, not a high-fidelity N-body run.',
        'Multiple moons can coexist, but crowded orbital zones are vulnerable to resonance overlap and close encounters.',
        'The visual focuses on orbital sorting, collisions, and survivor configurations.'
    ]
)


