import six

import bathy

##

six.moves.reload_module(bathy)

dem=bathy.dem()

zoom=(643000.,649200,
      4183000., 4188500)


tile=dem.extract_tile(zoom,res=2.0)

##

fig=plt.figure(1,figsize=(10,8))
fig.clf()

ax=fig.add_subplot(1,1,1)

img=tile.plot(ax=ax,cmap='jet')
img.set_clim([-10,5])
plt.colorbar(img)

fig.tight_layout()

fig.savefig('derived/bathy-snapshot.png',dpi=150)



