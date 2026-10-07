from cinematic_space_engine_v2 import *
# Output :https://www.youtube.com/shorts/Yrc2c-G4b2U

def draw(r, img, t, sh, p):
    c=(r.W*0.5, r.H*0.54)
    if sh.name == 'approach':
        sat=(r.W*0.5, r.H*0.60)
        r.planet(img, sat, r.W*0.12, base=(186,154,112), bands=True, rim=(255,220,180))
        r.ring_system(img, sat, r.W*0.41, r.W*0.10, density=1.1, particle=True, gap=True)
        r.text_center(img, 'FALLING THROUGH SATURN\'S RINGS', int(r.H*0.16), size=52)
    elif sh.name == 'gap_reveal':
        sat=(r.W*0.5, r.H*0.63)
        r.planet(img, sat, r.W*0.12, base=(186,154,112), bands=True, rim=(255,220,180))
        r.ring_system(img, sat, r.W*0.42, r.W*0.11, density=1.0, particle=True, gap=True)
        r.small_label(img, 'Cassini Division', int(r.W*0.68), int(r.H*0.34), color=(255,180,110))
        r.arrow(img, (r.W*0.77, r.H*0.39), (r.W*0.66, r.H*0.47), color=(255,180,110), width=4)
        r.text_center(img, 'THE RINGS ARE BROKEN INTO DISTINCT BANDS', int(r.H*0.16), size=42)
    elif sh.name == 'inside_stream':
        d=ImageDraw.Draw(img,'RGBA')
        cx, cy = r.W*0.5, r.H*0.5
        for i,(a, rr, z, size, phase) in enumerate(r.particle_bank[:420]):
            px = cx + math.sin(a*2+t*0.7+phase)*r.W*0.26*(0.15+rr)
            py = cy + (rr-0.5)*r.H*0.95 + math.cos(a+t+phase)*r.H*0.03
            rad = max(0.7, size*r.S*(1.1+0.5*p))
            alp = int(70+140*(1-abs(rr-0.52)))
            d.ellipse((px-rad, py-rad, px+rad, py+rad), fill=(238,230,214,alp))
        r.text_center(img, 'MOST OF THE RING IS EMPTY SPACE', int(r.H*0.15), size=48)
        r.hud(img, 'EXPERIENCE', 'flying through a particle stream', int(r.H*0.20))
    elif sh.name == 'particle_closeup':
        d=ImageDraw.Draw(img,'RGBA')
        for i in range(22):
            x = r.W*(0.12 + (i%5)*0.18 + 0.02*math.sin(t+i))
            y = r.H*(0.28 + (i//5)*0.12 + 0.02*math.cos(t*0.7+i))
            rr = r.W*(0.012 + 0.015*((i*3)%7)/6)
            d.ellipse((x-rr, y-rr*0.9, x+rr, y+rr*0.9), fill=(190+2*i, 180+2*i, 168+2*i, 230), outline=(250,244,235,140))
        r.text_center(img, 'ICE BLOCKS, PEBBLES, AND DUST', int(r.H*0.16), size=46)
    elif sh.name == 'collisions':
        d=ImageDraw.Draw(img,'RGBA')
        for i in range(180):
            x = r.W*fract(0.12*i + t*0.13)
            y = r.H*(0.15 + fract(0.07*i + t*0.21)*0.70)
            rr = r.W*(0.002 + 0.004*((i*5)%9)/8)
            d.ellipse((x-rr, y-rr, x+rr, y+rr), fill=(238,228,212,90))
        for j in range(5):
            x = r.W*(0.22 + j*0.13)
            y = r.H*(0.40 + 0.05*math.sin(t*0.9+j))
            r.glow_circle(img, (x,y), r.W*(0.008+0.003*j), (255,150,95), core=(255,230,210,255), blur=10)
        r.text_center(img, 'SMALL IMPACTS CONSTANTLY STIR THE SWARM', int(r.H*0.16), size=41)
        r.meter(img, int(r.W*0.16), int(r.H*0.22), int(r.W*0.68), 0.62, 'COLLISION RATE', color=(255,130,110))
    elif sh.name == 'shadow_band':
        sat=(r.W*0.5, r.H*0.63)
        r.planet(img, sat, r.W*0.12, base=(186,154,112), bands=True, rim=(255,220,180))
        r.ring_system(img, sat, r.W*0.42, r.W*0.11, density=1.0, particle=True, gap=True)
        d=ImageDraw.Draw(img,'RGBA')
        d.rectangle((0,r.H*0.55,r.W,r.H*0.63), fill=(12,12,18,60))
        r.text_center(img, 'AND THE WHOLE THING CASTS SHADOWS', int(r.H*0.16), size=44)
    else:
        r.text_center(img, 'IT WOULD FEEL LESS LIKE A WALL…', int(r.H*0.16), size=44)
        r.text_center(img, '…AND MORE LIKE A SPARSE, GLITTERING STORM.', int(r.H*0.22), size=40, fill=(255,210,170,255))

shots=[
    Shot('approach',0,8,'First, you would not hit a solid disc. Saturn’s rings are a vast collection of orbiting particles.'),
    Shot('gap_reveal',8,16,'As you approach, you would see bright bands, dark gaps, and famous structures like the Cassini Division.'),
    Shot('inside_stream',16,25,'Once inside, the strangest thing is how empty it feels. Most of the volume is just space.'),
    Shot('particle_closeup',25,34,'But around you, the ring material ranges from dust grains to icy chunks and larger blocks.'),
    Shot('collisions',34,43,'Those particles are constantly bumping, clumping, and shuffling energy between neighboring orbits.'),
    Shot('shadow_band',43,51,'The ring system also throws shadows across Saturn and across itself, carving dark lanes into the glow.'),
    Shot('outro',51,58,'So falling through Saturn’s rings would be spectacular — not a crash into a plate, but a trip through a sparse, glittering particle stream.')
]

spec=Spec(
    title='FALLING THROUGH SATURN\'S RINGS',
    subtitle='approach // particle stream // ring shadows',
    basename='falling_through_saturns_rings',
    shots=shots,
    draw=draw,
    notes=[
        'Saturn’s rings are extremely broad but vertically thin and mostly empty space.',
        'The ring material is primarily water ice ranging from tiny grains to larger chunks.',
        'This cinematic rendering emphasizes the subjective experience of passing through the ring plane.'
    ]
)


if __name__ == "__main__":
    run_cli(spec)

