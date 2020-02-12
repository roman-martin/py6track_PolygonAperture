import numpy as np
import pysixtrack
import py6track_PolygonAperture
import matplotlib.pyplot as plt
from mpmath import mp

mypolygon = np.array([  [	0.053833780684895	,	-0.030413209559333	],
                        [	0.003217156976424	,	-0.011239656066678	],
                        [	0.031957104005021	,	0.039669432352534	],
                        [	0.007506703165576	,	0.08	],
                        [	0.001501324285082	,	0.040330567647467	],
                        [	-0.027238624520475	,	0.046281004411289	],
                        [	-0.011367296512772	,	0.015206632168345	],
                        [	-0.050831093021608	,	-0.0204958514712	],
                        [	-0.052975866116184	,	0.017851228125433	],
                        [	-0.061554976264089	,	0.032396697610167	],
                        [	-0.08	,	-0.008595060109589	],
                        [	-0.063270791185829	,	-0.051570251102822	],
                        [	-0.014369985953019	,	-0.08	],
                        [	-0.027238624520475	,	-0.044958679044067	],
                        [	0.003217156976424	,	-0.041652893014689	],
                        [	0.013941022449304	,	-0.061487581802278	],
                        [	0.019946365790595	,	-0.035041320955933	],
                        [	0.049115278099867	,	-0.043636353676844	],
                        [	0.048686323480952	,	-0.079338837316389	],
                        [	0.05512063921076	,	-0.078016539337845	],
                        [	0.053833777130974	,	-0.042975218381911	],
                        [	0.08	,	-0.0171900928305	],
                        [	0.04353887693877	,	0.029090911580789	],
                        [	0.023378018734557	,	0.007272734742367	],
                        [	0.03024129086024	,	-0.003305786029378	],
                        [	0.054691684591845	,	-0.007933897425978	]]
                   ).transpose()

aper_elem = py6track_PolygonAperture.LimitPolygon(aperture = mypolygon)
N_part = 20000



#----Test scalar----------------------------------------
p_scalar = pysixtrack.Particles()
passed_particles_x = []
passed_particles_y = []
lost_particles_x = []
lost_particles_y = []
for n in range(N_part):
    p_scalar.x = (np.random.rand()-0.5) * 2.*8.5e-2
    p_scalar.y = (np.random.rand()-0.5) * 2.*8.5e-2
    p_scalar.state = 1

    aper_elem.track(p_scalar)
    if p_scalar.state == 1:
        passed_particles_x += [p_scalar.x]
        passed_particles_y += [p_scalar.y]
    else:
        lost_particles_x += [p_scalar.x]
        lost_particles_y += [p_scalar.y]


#----Test vector----------------------------------------
p_vec = pysixtrack.Particles()
p_vec.x = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
p_vec.y = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
p_vec.state = np.ones_like(p_vec.x, dtype=np.int)

aper_elem.track(p_vec)


#----Test mpmath compatibility--------------------------
p_mp = pysixtrack.Particles()
p_mp.x = mp.mpf(0.02)
p_mp.y = mp.mpf(0.01)
p_mp.state = 1
polygon_mp = [[mp.mpf(-0.03),mp.mpf(0.03),mp.mpf(0.03),mp.mpf(-0.03)],
              [mp.mpf(0.04),mp.mpf(0.04), mp.mpf(-0.04),mp.mpf(-0.04)]]
aper_elem_mp = py6track_PolygonAperture.LimitPolygon(aperture = polygon_mp)

aper_elem_mp.track(p_mp)
assert p_mp.state == 1
p_mp.x = mp.mpf(0.05)
p_mp.y = mp.mpf(0.01)
p_mp.state = 1
aper_elem_mp.track(p_mp)
assert p_mp.state == 0


#----Plots----------------------------------------------
fig, (ax_scalar, ax_vec) = plt.subplots(2,1)
ax_scalar.scatter(passed_particles_x, passed_particles_y, color='r', s=2)
ax_scalar.scatter(lost_particles_x, lost_particles_y, color='gray', s=2)
ax_vec.scatter(p_vec.x, p_vec.y, color='r', s=2)
ax_vec.scatter(p_vec.lost_particles[0].x, p_vec.lost_particles[0].y, color='gray', s=2)

ax_scalar.plot(mypolygon[0], mypolygon[1], color='k')
ax_scalar.plot([mypolygon[0,-1], mypolygon[0,0]], [mypolygon[1,-1],mypolygon[1,0]], color='k')
ax_scalar.set_title('Scalar')
ax_scalar.label_outer()
ax_vec.plot(mypolygon[0], mypolygon[1], color='k')
ax_vec.plot([mypolygon[0,-1], mypolygon[0,0]], [mypolygon[1,-1],mypolygon[1,0]], color='k')
ax_vec.set_title('Vector')

fig.savefig('polygonAperture_test.pdf', bbox_inches='tight')
