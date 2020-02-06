import numpy as np
import pysixtrack
import py6track_PolygonAperture
import matplotlib.pyplot as plt

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
p = pysixtrack.Particles()
p.x = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
p.y = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
p.state = np.ones_like(p.x, dtype=np.int)

aper_elem.track(p)


fig, ax = plt.subplots(1)
ax.scatter(p.x, p.y, color='r', s=2)
ax.scatter(p.lost_particles[0].x, p.lost_particles[0].y, color='gray', s=2)

ax.plot(mypolygon[0], mypolygon[1], color='k')
ax.plot([mypolygon[0,-1], mypolygon[0,0]], [mypolygon[1,-1],mypolygon[1,0]], color='k')

#fig.savefig('polygonAperture_test.pdf', bbox_inches='tight')
