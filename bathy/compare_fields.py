from stompy.spatial import field

import bathy
dem=bathy.dem()

##
clip=(646966, 647602, 4185504, 4186080)

tile=dem.extract_tile(clip)
##

txy00=['ADCP RBF cluster=3k med=3',field.GdalGrid('adcp-rbf-cluster-med3.tif')]
txy01=['DEM RBF qtracks',field.GdalGrid('tile-rbf-qtracks.tif')]

##

candidates=[txy00,txy01]
truth=tile

for i,cand in enumerate(candidates):
    fig=plt.figure(20+1)
    fig.clf()
    ax=fig.add_subplot(1,1,1)
    label,f = cand

    if (f.extents==truth.extents) and (f.dx==truth.dx) and (f.dy==truth.dy):
        delta=f.F - truth.F


