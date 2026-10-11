
from cinematic_space_engine_v2 import *


def draw(r, img, t, sh, p):
    earth = (r.W*0.5, r.H*0.47)
    if sh.name == 'scale_reveal':
        r.planet(img, earth, r.W*0.085, base=(45,120,205), rim=(120,220,255))
        r.ring_system(img, earth, r.W*0.40, r.W*0.10, density=1.0, gap=True, particle=True)
        r.text_center(img, 'A PLANET-SIZED HALO', int(r.H*0.18), size=54)
    elif sh.name == 'earth_crossing':
        r.planet(img, earth, r.W*0.088, base=(45,120,205), rim=(120,220,255))
        r.ring_system(img, earth, r.W*0.45, r.W*0.11, density=1.0, gap=True, particle=True)
        d=ImageDraw.Draw(img,'RGBA')
        d.line((r.W*0.08, r.H*0.47, r.W*0.92, r.H*0.47), fill=(255,210,170, 90), width=max(1,int(4*r.S)))
        r.text_center(img, 'THE RINGS WOULD CUT ACROSS THE SKY', int(r.H*0.18), size=42)
        r.hud(img, 'VISIBLE', 'a bright band from horizon to horizon', int(r.H*0.23))
    elif sh.name == 'from_ground':
        d=ImageDraw.Draw(img,'RGBA')
        d.polygon([(0,r.H*0.82),(r.W*0.15,r.H*0.73),(r.W*0.36,r.H*0.78),(r.W*0.56,r.H*0.68),(r.W*0.82,r.H*0.77),(r.W,r.H*0.70),(r.W,r.H),(0,r.H)], fill=(8,14,24,255))
        for i in range(7):
            yy = r.H*(0.30+0.03*i)
            d.arc((-r.W*0.15, yy-r.H*0.25, r.W*1.15, yy+r.H*0.12), 190, 350, fill=(236,230,215,120-10*i), width=max(1,int((7-i)*r.S)))
        r.text_center(img, 'NIGHTS WOULD LOOK UNREAL', int(r.H*0.17), size=48)
    elif sh.name == 'not_a_bridge':
        r.planet(img, earth, r.W*0.09, base=(45,120,205))
        r.ring_system(img, earth, r.W*0.43, r.W*0.11, density=1.0, gap=True, particle=True)
        r.arrow(img, (r.W*0.78, r.H*0.39), (r.W*0.62, r.H*0.45), color=(255,180,110), width=4)
        r.small_label(img, 'not solid', int(r.W*0.75), int(r.H*0.34), color=(255,180,110))
        r.text_center(img, 'THEY WOULD NOT BE A SOLID BRIDGE', int(r.H*0.18), size=44)
    elif sh.name == 'shear':
        r.planet(img, earth, r.W*0.09, base=(45,120,205))
        r.ring_system(img, earth, r.W*0.43, r.W*0.11, density=1.0, gap=True, particle=True)
        for j,x in enumerate([0.24,0.40,0.56,0.72]):
            r.arrow(img, (r.W*x, r.H*0.40), (r.W*x+(j-1.5)*24*r.S, r.H*0.31), color=(100,230,255), width=3)
        r.meter(img, int(r.W*0.16), int(r.H*0.20), int(r.W*0.68), 0.76, 'SHEAR + COLLISIONS', color=(255,130,110))
        r.text_center(img, 'EACH RING PARTICLE ORBITS ON ITS OWN', int(r.H*0.18), size=40)
    elif sh.name == 'storms':
        d=ImageDraw.Draw(img,'RGBA')
        d.polygon([(0,r.H*0.82),(r.W*0.18,r.H*0.76),(r.W*0.42,r.H*0.81),(r.W*0.71,r.H*0.74),(r.W,r.H*0.79),(r.W,r.H),(0,r.H)], fill=(5,11,20,255))
        for i in range(18):
            x = r.W*(0.08 + i*0.05)
            d.line((x, r.H*0.62, x-r.W*0.02, r.H*0.80), fill=(180,210,255, 60), width=max(1,int(2*r.S)))
        for i in range(6):
            yy = r.H*(0.28+0.03*i)
            d.arc((-r.W*0.2, yy-r.H*0.25, r.W*1.2, yy+r.H*0.12), 190, 350, fill=(236,230,215,90-8*i), width=max(1,int((6-i)*r.S)))
        r.text_center(img, 'SHADOWS, GLARE, METEOR DUST', int(r.H*0.17), size=44)
    else:
        r.text_center(img, 'BEAUTIFUL — BUT MESSY PHYSICS', int(r.H*0.18), size=46)
        r.hud(img, 'BOTTOM LINE', 'gorgeous sky, unstable details', int(r.H*0.23))

shots=[
    Shot('scale_reveal',0,8,'If Earth had rings as wide as Saturn’s, our planet would wear a gigantic, glowing halo.'),
    Shot('earth_crossing',8,16,'From orbit, the rings would stretch far beyond the globe and cut straight across the equator.'),
    Shot('from_ground',16,24,'From the ground, they could appear as an enormous luminous band arching from horizon to horizon.'),
    Shot('not_a_bridge',24,32,'But rings are not solid surfaces. They are made of countless icy particles, chunks, and dust.'),
    Shot('shear',32,41,'Each particle follows its own orbit, so the ring is constantly shearing, colliding, and rearranging itself.'),
    Shot('storms',41,50,'That would create dramatic shadows, intense glare, and a sky full of extra debris effects.'),
    Shot('outro',50,58,'So Saturn-like rings around Earth would be visually spectacular — but they would be a dynamic swarm, not a giant cosmic bridge.')
]

spec=Spec(
    title='WHAT IF SATURN\'S RINGS REACHED EARTH?',
    subtitle='scale reveal // from the ground // ring particle reality',
    basename='what_if_saturns_rings_reached_earth',
    shots=shots,
    draw=draw,
    notes=[
        'This is a cinematic scale thought experiment, not an exact orbital engineering proposal.',
        'Saturn’s rings are made of enormous numbers of particles over a very thin vertical thickness.',
        'The visuals emphasize sky appearance, ring geometry, and the fact that rings are particulate rather than solid.'
    ]
)

