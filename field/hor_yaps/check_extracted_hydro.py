# extract_velocities_3.py can now pull from ptm average files.
# but does it do so correctly?
# maybe.  not obviously wrong.

import track_common
import matplotlib.pyplot as plt
import stompy.plot.cmap as scmap
from stompy.spatial import field

## 
turbo=scmap.load_gradient('turbo.cpt')
dem=field.GdalGrid("../../bathy/junction-composite-dem-no_adcp.tif")

##

tracks=track_common.read_from_folder('mergedhydro_v00')
##

track_lengths=[ len(t) for t in tracks['withhydro']]

longest=np.argmax(track_lengths)

track=tracks.iloc[38,:]['withhydro']

##

fig,ax=plt.subplots(1,1,num=1)

ax.plot(track.x,track.y,'g.')

pad=50
zoom=[ track.x.min()-pad,
       track.x.max()+pad,
       track.y.min()-pad,
       track.y.max()+pad ]

dc=dem.crop([ zoom[0]-500,
              zoom[1]+500,
              zoom[2]-500,
              zoom[3]+500] )
dc.plot(ax=ax,zorder=-5,cmap='gray',clim=[-20,10],interpolation='bilinear')
dc.plot_hillshade(ax=ax,z_factor=1,plot_args=dict(interpolation='bilinear'))
ax.set_adjustable('datalim')
ax.axis(zoom)
ax.axis('off')

qset=ax.quiver( track.x, track.y, track.model_u, track.model_v, color='blue')

ax.quiverkey(qset,0.1,0.1,0.25,label="0.25 m/s",coordinates='axes')

##

plt.figure(2).clf()
fig,axs=plt.subplots(2,1,sharex=True,num=2)
track['model_u_mag']=np.sqrt( track.model_u**2 + track.model_v**2 )
track['model_u_surf_mag']=np.sqrt( track.model_u_surf**2 + track.model_v_surf**2 )

axs[0].hist(track['model_u_mag'],bins=50)
axs[1].hist(track['model_u_surf_mag'],bins=50)

