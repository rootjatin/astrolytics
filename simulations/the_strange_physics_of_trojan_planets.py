from cinematic_space_engine_v2 import *

output : https://youtube.com/shorts/HiJMSS9Wo00?feature=share
def draw(r, img, t, sh, p):
    c = (r.W*0.5, r.H*0.43)
    R = r.W*0.30
    giant_a = -0.6 + t*0.15
    trojan_offset = math.pi/3
    giant = (c[0]+math.cos(giant_a)*R, c[1]+math.sin(giant_a)*R*0.36)
    l4 = (c[0]+math.cos(giant_a+trojan_offset)*R, c[1]+math.sin(giant_a+trojan_offset)*R*0.36)
    l5 = (c[0]+math.cos(giant_a-trojan_offset)*R, c[1]+math.sin(giant_a-trojan_offset)*R*0.36)
    if sh.name == 'hook':
        r.glow_circle(img, c, r.W*0.042, (255,190,80), core=(255,248,220,255), blur=33)
        r.orbit(img, c, R, R*0.36, alpha=100, width=3)
        r.planet(img, giant, r.W*0.038, base=(210,150,85), bands=True, rim=(255,220,170))
        r.glow_circle(img, l4, r.W*0.015, (100,235,255), core=(180,245,255,255), blur=10)
        r.glow_circle(img, l5, r.W*0.015, (160,130,255), core=(220,210,255,255), blur=10)
        r.text_center(img, 'WORLDS HIDING IN LAGRANGE POINTS', int(r.H*0.18), size=48)
    elif sh.name == 'lagrange_map':
        r.glow_circle(img, c, r.W*0.042, (255,190,80), core=(255,248,220,255), blur=33)
        r.orbit(img, c, R, R*0.36, alpha=100, width=3)
        r.planet(img, giant, r.W*0.040, base=(205,145,88), bands=True, rim=(255,220,170))
        for pt, label, col in [(l4, 'L4', (95,230,255)), (l5, 'L5', (165,135,255))]:
            r.glow_circle(img, pt, r.W*0.017, col, core=(*col,255), blur=12)
            r.small_label(img, label, int(pt[0]+18*r.S), int(pt[1]-18*r.S), color=col)
        d = ImageDraw.Draw(img,'RGBA')
        d.line([giant, l4], fill=(220,240,255,70), width=max(1,int(2*r.S)))
        d.line([giant, c], fill=(220,240,255,70), width=max(1,int(2*r.S)))
        r.text_center(img, 'L4 AND L5 ARE 60° AHEAD OR BEHIND', int(r.H*0.20), size=42)
    elif sh.name == 'tadpole_motion':
        r.glow_circle(img, c, r.W*0.041, (255,190,80), core=(255,248,220,255), blur=30)
        r.orbit(img, c, R, R*0.36, alpha=90, width=2)
        r.planet(img, giant, r.W*0.040, base=(210,150,88), bands=True)
        pts=[]
        for k in range(42):
            aa = giant_a+trojan_offset+math.sin((t-k*0.05)*0.7)*0.24
            pts.append((c[0]+math.cos(aa)*R, c[1]+math.sin(aa)*R*0.36))
        r.trail(img, pts, (95,230,255), 5)
        r.glow_circle(img, pts[-1], r.W*0.017, (95,230,255), core=(190,248,255,255), blur=10)
        r.hud(img, 'TROJAN MODE', 'tadpole libration', int(r.H*0.17))
    elif sh.name == 'horseshoe':
        r.draw_grid(img, r.W*0.10, r.H*0.26, r.W*0.80, r.H*0.34, rows=4, cols=4)
        pts=[]
        for i in range(240):
            q = i/239
            ang = q*math.tau
            rr = 0.72 - 0.26*math.cos(ang)
            x = r.W*0.50 + math.sin(ang)*r.W*0.22*rr
            y = r.H*0.43 - math.cos(ang)*r.H*0.13
            pts.append((x,y))
        r.trail(img, pts, (255,180,110), 4)
        r.glow_circle(img, pts[int((0.25+0.5*p)*239)], r.W*0.015, (255,180,110), core=(255,236,205,255), blur=8)
        r.text_center(img, 'SOME CO-ORBITALS TRACE HORSESHOES', int(r.H*0.18), size=44)
    elif sh.name == 'capture':
        r.radar(img, (r.W*0.5, r.H*0.43), r.W*0.20)
        for i in range(36):
            ang = i/36*math.tau
            rr = r.W*(0.04+0.14*(i%7)/6)
            pt = (r.W*0.5+math.cos(ang+t*0.2)*rr, r.H*0.43+math.sin(ang+t*0.2)*rr)
            col = (110,230,255) if i%3 else (255,180,110)
            r.glow_circle(img, pt, r.W*0.006, col, core=(*col,255), blur=6)
        r.meter(img, int(r.W*0.16), int(r.H*0.66), int(r.W*0.68), 0.55+0.25*math.sin(t*0.6), 'CAPTURE WINDOW', color=(100,235,255))
        r.text_center(img, 'MIGRATION CAN TRAP BODIES THERE', int(r.H*0.18), size=45)
    elif sh.name == 'exoworlds':
        r.glow_circle(img, c, r.W*0.041, (255,190,80), core=(255,248,220,255), blur=30)
        for j,off in enumerate([0, trojan_offset, -trojan_offset]):
            pt = (c[0]+math.cos(giant_a+off)*R, c[1]+math.sin(giant_a+off)*R*0.36)
            if j==0:
                r.planet(img, pt, r.W*0.038, base=(206,146,88), bands=True)
            else:
                r.planet(img, pt, r.W*0.019, base=((70,210,255) if j==1 else (170,130,255)), rim=(200,240,255))
        r.text_center(img, 'TROJAN PLANETS COULD EXIST AROUND OTHER STARS', int(r.H*0.18), size=40)
    else:
        r.text_center(img, 'SAME ORBIT. DIFFERENT SAFE ZONES.', int(r.H*0.18), size=44)
        r.hud(img, 'BIG IDEA', 'gravity can create parking spots', int(r.H*0.22))

shots = [
    Shot('hook', 0, 7, 'Trojan planets would share an orbit with a larger world — not by colliding with it, but by living in gravitational safe zones.'),
    Shot('lagrange_map', 7, 15, 'Those safe zones are the L4 and L5 Lagrange points, about 60 degrees ahead of or behind the main planet.'),
    Shot('tadpole_motion', 15, 24, 'A Trojan does not sit perfectly still. It usually librates, tracing a tadpole-shaped path around one of those points.'),
    Shot('horseshoe', 24, 33, 'Other co-orbitals can trace wider horseshoe paths, trading orbital energy while avoiding direct collision.'),
    Shot('capture', 33, 42, 'Migration through a young disk could let asteroids, moons, or even planets get captured into these resonant locations.'),
    Shot('exoworlds', 42, 51, 'That means exoplanet systems might hide Trojan companions that are hard to spot because they keep the same year as a larger planet.'),
    Shot('outro', 51, 58, 'Trojan physics is strange because the orbit is shared — but the gravity landscape makes different parts of that orbit behave very differently.')
]

spec = Spec(
    title='THE STRANGE PHYSICS OF TROJAN PLANETS',
    subtitle='L4 // L5 // tadpole orbits // horseshoe motion',
    basename='the_strange_physics_of_trojan_planets',
    shots=shots,
    draw=draw,
    notes=[
        'The L4 and L5 points are equilibrium regions in the circular restricted three-body problem.',
        'Real Trojan behavior can include libration rather than perfect station keeping.',
        'This video illustrates gravitational geometry rather than claiming a confirmed Trojan Earth analogue.'
    ]
)




if __name__ == "__main__":
    run_cli(spec)

