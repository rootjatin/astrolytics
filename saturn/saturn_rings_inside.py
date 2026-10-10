from cinematic_space_engine_v2 import *
# https://www.youtube.com/shorts/3BJjycdkVAw

def draw(r, img, t, sh, p):
    sat=(r.W*0.5, r.H*0.54)
    if sh.name == 'macro':
        r.planet(img, sat, r.W*0.12, base=(185,152,111), bands=True, rim=(255,220,180))
        r.ring_system(img, sat, r.W*0.40, r.W*0.10, density=1.1, particle=True, gap=True)
        r.text_center(img, 'WHAT HAPPENS INSIDE A PLANETARY RING?', int(r.H*0.16), size=49)
    elif sh.name == 'orbits':
        r.planet(img, sat, r.W*0.11, base=(185,152,111), bands=True)
        for k in range(5):
            r.orbit(img, sat, r.W*(0.22+0.045*k), r.W*(0.06+0.012*k), alpha=75, width=2, dash=(k%2==0))
        for j in range(14):
            a = j/14*math.tau + t*0.45*(1+0.03*j)
            rr = r.W*(0.23+0.18*(j%5)/4)
            pt=(sat[0]+math.cos(a)*rr, sat[1]+math.sin(a)*rr*(0.27+0.02*(j%4)))
            r.glow_circle(img, pt, r.W*0.005, (238,230,214), core=(255,244,228,255), blur=4)
        r.text_center(img, 'EVERY PARTICLE KEEPS ITS OWN ORBIT', int(r.H*0.16), size=44)
    elif sh.name == 'shear':
        d=ImageDraw.Draw(img,'RGBA')
        for i in range(7):
            y = r.H*(0.26+0.07*i)
            d.line((r.W*0.12,y,r.W*0.88,y), fill=(120,190,255,35), width=max(1,int(2*r.S)))
        for i in range(80):
            lane=i%7
            x = r.W*(0.15 + fract(0.10*i + t*(0.04+lane*0.01))*0.70)
            y = r.H*(0.26+0.07*lane + 0.01*math.sin(i+t))
            rr = r.W*(0.003+0.002*(i%5)/4)
            d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=(238,230,214,110))
        r.text_center(img, 'INNER PARTS MOVE FASTER THAN OUTER PARTS', int(r.H*0.16), size=42)
        r.arrow(img, (r.W*0.22,r.H*0.29), (r.W*0.42,r.H*0.29), color=(100,230,255), width=3)
        r.arrow(img, (r.W*0.22,r.H*0.64), (r.W*0.33,r.H*0.64), color=(255,180,110), width=3)
    elif sh.name == 'wakes':
        d=ImageDraw.Draw(img,'RGBA')
        for i in range(220):
            q = i/219
            x = r.W*0.12+r.W*0.76*q
            y = r.H*0.48 + math.sin(q*math.tau*6+t*0.9)*r.H*0.05*(1-q*0.2)
            rr = r.W*(0.002+0.003*((i*3)%7)/6)
            d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=(235,228,212,120))
        r.trail(img, [(r.W*0.20+i*r.W*0.012, r.H*0.48+math.sin(i*0.25+t)*r.H*0.05) for i in range(45)], (95,230,255), 4)
        r.text_center(img, 'GRAVITY CAN SCULPT WAVES AND WAKES', int(r.H*0.16), size=44)
    elif sh.name == 'moonlets':
        d=ImageDraw.Draw(img,'RGBA')
        center=(r.W*0.5,r.H*0.48)
        for i in range(300):
            a = i/300*math.tau
            rr = r.W*(0.16+0.20*((i*7)%100)/100)
            px = center[0]+math.cos(a)*rr
            py = center[1]+math.sin(a)*rr*0.25
            rad = r.W*(0.002+0.002*(i%4)/3)
            d.ellipse((px-rad,py-rad,px+rad,py+rad), fill=(238,230,214,80))
        moon = (r.W*0.55, r.H*0.48)
        r.glow_circle(img, moon, r.W*0.012, (190,210,255), core=(240,245,255,255), blur=6)
        r.arrow(img, (r.W*0.63, r.H*0.42), moon, color=(255,180,110), width=4)
        r.small_label(img, 'moonlet', int(r.W*0.64), int(r.H*0.38), color=(255,180,110))
        r.text_center(img, 'TINY MOONS CAN OPEN GAPS AND DISTURB STREAMS', int(r.H*0.16), size=39)
    elif sh.name == 'self_gravity':
        r.radar(img, (r.W*0.5,r.H*0.46), r.W*0.22)
        for i in range(48):
            ang = i/48*math.tau + t*0.1
            rr = r.W*(0.05+0.18*((i*7)%19)/18)
            pt=(r.W*0.5+math.cos(ang)*rr, r.H*0.46+math.sin(ang)*rr)
            col = (95,230,255) if i%2==0 else (240,230,214)
            r.glow_circle(img, pt, r.W*0.0045, col, core=(*col,255), blur=4)
        r.text_center(img, 'THE RING IS A WHOLE ECOSYSTEM OF MICRO-PHYSICS', int(r.H*0.16), size=40)
    else:
        r.text_center(img, 'A PLANETARY RING ISN\'T STATIC.', int(r.H*0.16), size=46)
        r.text_center(img, 'IT\'S A LIVING SWARM OF ORBITS, COLLISIONS, AND WAVES.', int(r.H*0.22), size=36, fill=(255,210,170,255))

shots=[
    Shot('macro',0,7,'A planetary ring looks smooth from afar, but inside it is a crowded swarm of individual orbiting particles.'),
    Shot('orbits',7,15,'Each particle follows its own path around the planet, all at slightly different distances and speeds.'),
    Shot('shear',15,24,'That speed difference creates shear: inner ring material laps outer material over and over again.'),
    Shot('wakes',24,33,'Gravitational disturbances can then sculpt ripples, density waves, and spiral-like wakes through the ring.'),
    Shot('moonlets',33,42,'Small embedded moonlets can open gaps, kick particles aside, and carve out local structure.'),
    Shot('self_gravity',42,50,'Collisions, self-gravity, and repeated stirring keep the ring in a constant state of rearrangement.'),
    Shot('outro',50,58,'So inside a ring system, nothing is frozen in place. It is a kinetic, ever-changing ecosystem of orbiting debris.')
]

spec=Spec(
    title='WHAT HAPPENS INSIDE A PLANETARY RING?',
    subtitle='shear // waves // moonlets // self-gravity',
    basename='what_happens_inside_a_planetary_ring',
    shots=shots,
    draw=draw,
    notes=[
        'The video focuses on the micro-physics of particulate ring systems.',
        'Real ring dynamics include differential rotation, collisional damping, density waves, wakes, and perturbations from moons.',
        'This is a simplified visual explanation rather than a numerical astrophysics paper.'
    ]
)

from cinematic_space_engine_v2 import *


def draw(r, img, t, sh, p):
    sat=(r.W*0.5, r.H*0.54)
    if sh.name == 'macro':
        r.planet(img, sat, r.W*0.12, base=(185,152,111), bands=True, rim=(255,220,180))
        r.ring_system(img, sat, r.W*0.40, r.W*0.10, density=1.1, particle=True, gap=True)
        r.text_center(img, 'WHAT HAPPENS INSIDE A PLANETARY RING?', int(r.H*0.16), size=49)
    elif sh.name == 'orbits':
        r.planet(img, sat, r.W*0.11, base=(185,152,111), bands=True)
        for k in range(5):
            r.orbit(img, sat, r.W*(0.22+0.045*k), r.W*(0.06+0.012*k), alpha=75, width=2, dash=(k%2==0))
        for j in range(14):
            a = j/14*math.tau + t*0.45*(1+0.03*j)
            rr = r.W*(0.23+0.18*(j%5)/4)
            pt=(sat[0]+math.cos(a)*rr, sat[1]+math.sin(a)*rr*(0.27+0.02*(j%4)))
            r.glow_circle(img, pt, r.W*0.005, (238,230,214), core=(255,244,228,255), blur=4)
        r.text_center(img, 'EVERY PARTICLE KEEPS ITS OWN ORBIT', int(r.H*0.16), size=44)
    elif sh.name == 'shear':
        d=ImageDraw.Draw(img,'RGBA')
        for i in range(7):
            y = r.H*(0.26+0.07*i)
            d.line((r.W*0.12,y,r.W*0.88,y), fill=(120,190,255,35), width=max(1,int(2*r.S)))
        for i in range(80):
            lane=i%7
            x = r.W*(0.15 + fract(0.10*i + t*(0.04+lane*0.01))*0.70)
            y = r.H*(0.26+0.07*lane + 0.01*math.sin(i+t))
            rr = r.W*(0.003+0.002*(i%5)/4)
            d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=(238,230,214,110))
        r.text_center(img, 'INNER PARTS MOVE FASTER THAN OUTER PARTS', int(r.H*0.16), size=42)
        r.arrow(img, (r.W*0.22,r.H*0.29), (r.W*0.42,r.H*0.29), color=(100,230,255), width=3)
        r.arrow(img, (r.W*0.22,r.H*0.64), (r.W*0.33,r.H*0.64), color=(255,180,110), width=3)
    elif sh.name == 'wakes':
        d=ImageDraw.Draw(img,'RGBA')
        for i in range(220):
            q = i/219
            x = r.W*0.12+r.W*0.76*q
            y = r.H*0.48 + math.sin(q*math.tau*6+t*0.9)*r.H*0.05*(1-q*0.2)
            rr = r.W*(0.002+0.003*((i*3)%7)/6)
            d.ellipse((x-rr,y-rr,x+rr,y+rr), fill=(235,228,212,120))
        r.trail(img, [(r.W*0.20+i*r.W*0.012, r.H*0.48+math.sin(i*0.25+t)*r.H*0.05) for i in range(45)], (95,230,255), 4)
        r.text_center(img, 'GRAVITY CAN SCULPT WAVES AND WAKES', int(r.H*0.16), size=44)
    elif sh.name == 'moonlets':
        d=ImageDraw.Draw(img,'RGBA')
        center=(r.W*0.5,r.H*0.48)
        for i in range(300):
            a = i/300*math.tau
            rr = r.W*(0.16+0.20*((i*7)%100)/100)
            px = center[0]+math.cos(a)*rr
            py = center[1]+math.sin(a)*rr*0.25
            rad = r.W*(0.002+0.002*(i%4)/3)
            d.ellipse((px-rad,py-rad,px+rad,py+rad), fill=(238,230,214,80))
        moon = (r.W*0.55, r.H*0.48)
        r.glow_circle(img, moon, r.W*0.012, (190,210,255), core=(240,245,255,255), blur=6)
        r.arrow(img, (r.W*0.63, r.H*0.42), moon, color=(255,180,110), width=4)
        r.small_label(img, 'moonlet', int(r.W*0.64), int(r.H*0.38), color=(255,180,110))
        r.text_center(img, 'TINY MOONS CAN OPEN GAPS AND DISTURB STREAMS', int(r.H*0.16), size=39)
    elif sh.name == 'self_gravity':
        r.radar(img, (r.W*0.5,r.H*0.46), r.W*0.22)
        for i in range(48):
            ang = i/48*math.tau + t*0.1
            rr = r.W*(0.05+0.18*((i*7)%19)/18)
            pt=(r.W*0.5+math.cos(ang)*rr, r.H*0.46+math.sin(ang)*rr)
            col = (95,230,255) if i%2==0 else (240,230,214)
            r.glow_circle(img, pt, r.W*0.0045, col, core=(*col,255), blur=4)
        r.text_center(img, 'THE RING IS A WHOLE ECOSYSTEM OF MICRO-PHYSICS', int(r.H*0.16), size=40)
    else:
        r.text_center(img, 'A PLANETARY RING ISN\'T STATIC.', int(r.H*0.16), size=46)
        r.text_center(img, 'IT\'S A LIVING SWARM OF ORBITS, COLLISIONS, AND WAVES.', int(r.H*0.22), size=36, fill=(255,210,170,255))

shots=[
    Shot('macro',0,7,'A planetary ring looks smooth from afar, but inside it is a crowded swarm of individual orbiting particles.'),
    Shot('orbits',7,15,'Each particle follows its own path around the planet, all at slightly different distances and speeds.'),
    Shot('shear',15,24,'That speed difference creates shear: inner ring material laps outer material over and over again.'),
    Shot('wakes',24,33,'Gravitational disturbances can then sculpt ripples, density waves, and spiral-like wakes through the ring.'),
    Shot('moonlets',33,42,'Small embedded moonlets can open gaps, kick particles aside, and carve out local structure.'),
    Shot('self_gravity',42,50,'Collisions, self-gravity, and repeated stirring keep the ring in a constant state of rearrangement.'),
    Shot('outro',50,58,'So inside a ring system, nothing is frozen in place. It is a kinetic, ever-changing ecosystem of orbiting debris.')
]

spec=Spec(
    title='WHAT HAPPENS INSIDE A PLANETARY RING?',
    subtitle='shear // waves // moonlets // self-gravity',
    basename='what_happens_inside_a_planetary_ring',
    shots=shots,
    draw=draw,
    notes=[
        'The video focuses on the micro-physics of particulate ring systems.',
        'Real ring dynamics include differential rotation, collisional damping, density waves, wakes, and perturbations from moons.',
        'This is a simplified visual explanation rather than a numerical astrophysics paper.'
    ]
)


if __name__ == "__main__":
    run_cli(spec)

