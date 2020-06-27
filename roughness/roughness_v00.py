from stompy.spatial import field
import matplotlib.pyplot as plt
import numpy as np
from stompy import utils
##

total_crop=[6321753,6325212,
            2116503,2118419]
# that covers the junction
dem=field.GdalGrid("../bathy/dwr/OldRiver_SJ_072020171.tif",
                   geo_bounds=total_crop)
dem.F[ dem.F<-1e+10 ]=np.nan

##

# that's in ft
# lowpass in 2D at 15ft
dx=20
dem_lp=dem.copy()
dem_lp.smooth_by_convolution(dx)
dem_hp=field.SimpleGrid(extents=dem.extents,F=dem.F-dem_lp.F)
dem_rms=dem_hp.copy()
dem_rms.F=dem_rms.F**2
dem_rms.smooth_by_convolution(dx)
dem_rms.F=np.sqrt(dem_rms.F)

## 

plt.figure(1).clf()
fig,axs=plt.subplots(1,4,sharex=True,sharey=True,num=1)

img1=dem.plot(vmin=-20,vmax=5,cmap='jet',ax=axs[0])
img2=dem_lp.plot(vmin=-20,vmax=5,cmap='jet',ax=axs[1])

img3=dem_hp.plot(vmin=-1,vmax=1,cmap='seismic',ax=axs[2])
img4=dem_rms.plot(vmin=0.1,vmax=1.0,cmap='plasma_r',ax=axs[3])

imgs=[img1,img2,img3,img4]
descs=["DEM","DEM lowpass","DEM highpass","DEM rms"]
for img,ax,desc in zip(imgs,axs,descs):
    plt.colorbar(img,ax=ax,orientation='horizontal',label=desc)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

zoom=(6322685., 6324276, 2115808, 2118406.)
axs[0].axis(zoom)
fig.tight_layout()

fig.savefig('roughness-dx%d.png'%dx)

##

# similar but on the grid
from stompy.grid import unstructured_grid
from stompy.spatial import proj_utils

g=unstructured_grid.UnstructuredGrid.from_ugrid("../model/suntans/grid-with_bathy.nc")
# reproject grid to 1ft dem
to_ca=proj_utils.mapper('EPSG:26910','EPSG:2227')
g.nodes['x'][:] = to_ca(g.nodes['x'])
g.cells_center(refresh=True)

## 

#ripple_xy=[6323641, 2116848.]
#riprap_xy=[6323582, 2116749]
cell_mean=np.zeros(g.Ncells(),np.float64)
cell_rms=np.zeros(g.Ncells(),np.float64)
X,Y=dem.XY()

#crop=(6323542, 6323719, 2116656, 2116927)
crop=total_crop

for c in utils.progress(np.nonzero(g.cell_clip_mask(crop))[0]):
    cell_poly=g.cell_polygon(c)
    try:
        sel=dem.polygon_mask(cell_poly)
    except AttributeError:
        print("!")
        continue
    sel=sel&np.isfinite(dem.F)
    if sel.sum()<10: # ad hoc
        continue

    dem_vals=dem.F[sel]
    dem_x=X[sel]
    dem_y=Y[sel]
    dem_x_loc =dem_x - dem_x.mean()
    dem_y_loc =dem_y - dem_y.mean()

    fit_y = dem_vals
    fit_X = np.c_[dem_x_loc, dem_y_loc, np.ones_like(dem_x)]
    beta_hat = np.linalg.lstsq(fit_X,fit_y,rcond=None)[0]
    flat=np.dot(fit_X,beta_hat)

    anom=dem_vals-flat
    cell_rms[c]=np.sqrt((anom**2).mean())

##

plt.figure(1).clf()
g.plot_edges(lw=0.1,color='k')
coll=g.plot_cells(values=cell_rms,cmap='plasma',mask=cell_rms!=0.0)
coll.set_clim([0,0.7])
plt.axis('equal')
plt.axis( (6322822., 6324505, 2116228, 2117796))
plt.colorbar(coll,label='RMS anomaly (ft)')

##

# Silly 2D FFT
zoom_riprap=(6323599.815442312, 6323667.380415566, 2116704.778420842, 2116755.125223493)
zoom_ripple=(6323686.314954443, 6323753.879927698, 2116777.519742855, 2116827.8665455063)

dem_riprap=dem.crop(zoom_riprap)
dem_ripple=dem.crop(zoom_ripple)

from scipy import fftpack, ndimage

fft2_riprap=fftpack.fft2(dem_riprap.F)
fft2_ripple=fftpack.fft2(dem_ripple.F)

plt.figure(3).clf()
fig,axs=plt.subplots(1,2,num=3,sharex=True,sharey=True)
img1=axs[0].imshow( np.log(np.abs(fft2_riprap)) )
img2=axs[1].imshow( np.log(np.abs(fft2_ripple)) )

plt.colorbar(img1,orientation='horizontal',ax=axs[0])
plt.colorbar(img2,orientation='horizontal',ax=axs[1])

plt.setp([img1,img2],clim=[0,4],cmap='jet')
