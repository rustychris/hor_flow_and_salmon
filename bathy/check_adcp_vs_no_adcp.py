"""
Revisiting after the fact --

is there anything to be gained from pulling in the ADCP data?

"""

from stompy.spatial import proj_utils, wkb2shp, field
import matplotlib.pyplot as plt
import xarray as xr

##

dem=field.GdalGrid('junction-composite-20190117-no_adcp.tif')

dem_adcp=field.GdalGrid('junction-composite-dem.tif')
## 

adcp_data=wkb2shp.shp2geom('derived/samples-depth.shp',target_srs='EPSG:26910')
##

adcp_ds=xr.Dataset()
adcp_ds['z_bed']=('sample',),adcp_data['depth']
pnts=np.array([np.array(g) for g in adcp_data['geom']])

adcp_ds['x']=('sample',), pnts[:,0]
adcp_ds['y']=('sample',), pnts[:,1]

##

zoom=(647088.821657404, 647702.7390601198, 4185484.2206095303, 4185941.6880934904)

plt.figure(10).clf()

fig,ax=plt.subplots(num=10)

dem_at_adcp=dem( np.c_[adcp_ds.x,adcp_ds.y] )
delta=dem_at_adcp - adcp_ds.z_bed

scat=ax.scatter(adcp_ds.x,adcp_ds.y,20,delta,cmap='coolwarm')
scat.set_clim([-0.75,0.75])

plt.colorbar(scat,ax=ax,orientation='horizontal',label='dem - adcp')

ax.axis('equal')
ax.axis(zoom)

##

# And difference between the ADCP interpolated bathy 
dem_delta=field.SimpleGrid(extents=dem_adcp.extents,F=dem_adcp.F-dem.F)

plt.figure(11).clf()
fig,ax=plt.subplots(num=11)
img=dem_delta.plot(cmap='coolwarm',clim=[-1,1])
plt.colorbar(img,ax=ax,orientation='horizontal',label='dem_adcp - dem_dwr')

ax.axis('equal')
ax.axis(zoom)
