"""
Study area figure
  Inset location in map of California - done-ish
  Model domain extent - done
  Bathymetry - done
  Color scale - done
  Receiver location - done
  Legend - done
  Distance scale - done
"""
from stompy import utils
from stompy.spatial import field, wkb2shp, proj_utils
import stompy.plot.cmap as scmap
from stompy.plot import plot_wkb, plot_utils
from stompy.grid import unstructured_grid
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from shapely import ops
##

upper_ll=[-120.930408, 37.309940]
lower_ll=[-121.271098, 37.691794]
release_xy=proj_utils.mapper('WGS84','EPSG:26910')( [upper_ll,lower_ll] )

    

##

dem_mrg=field.MultiRasterField(["../../../bathy/junction-composite-20200605-w_smooth.tif",
                                "/home/rusty/data/bathy_dwr/gtiff/dem_bay_delta_10m_v3_20121109_2.tif"])
clip=(646390., 648336., 4184677., 4187210.)
dem=dem_mrg.extract_tile(clip)

##

aerial=field.GdalGrid("../../../gis/aerial/m_3712114_sw_10_h_20160621_20161004-UTM.tif")
aerial.F=aerial.F[:,:,:3] # drop alpha - seems mis-scaled

##

#grid=unstructured_grid.UnstructuredGrid.read_ugrid("../../../model/grid/snubby_junction/snubby-06.nc")
grid=unstructured_grid.UnstructuredGrid.read_ugrid("../../../model/suntans/snubby-08-edit60-with_bathysmooth.nc")
grid_poly=grid.boundary_polygon()

##

rx_locs=pd.read_csv("../../../field/hor_yaps/yap-positions.csv")

##

ca_shp=wkb2shp.shp2geom("../../../gis/CA_State/CA_State_TIGER2016.shp",
                        target_srs='EPSG:26910')

##

sfe_poly=wkb2shp.shp2geom("../../../gis/dem_to_shoreline/shoreline_from_dem.shp")

# Extend further into ocean
ca_mpoly=ca_shp['geom'][0].buffer(-5556)
ocean_poly=ca_mpoly.envelope.difference(ca_mpoly)

sfe_geom = sfe_poly['geom'][0].union(ocean_poly)

##

# Extend SJ River
ca_rivers=wkb2shp.shp2geom("../../../gis/CA_State/hydro/CaliforniaHydro.shp",
                           target_srs='EPSG:26910')
sel=(ca_rivers['NAME']=='San Joaquin River') # |(ca_rivers['NAME']=='Sacramento River')
river_geoms=ca_rivers['geom'][sel]

river_polys=[g.buffer(50.0,resolution=4) for g in river_geoms]
river_poly=ops.cascaded_union(river_polys)

sfe_geom=sfe_geom.union( river_poly )

##

zoom=(646818.0, 648074.0, 4185291.632, 4186257.632)

fig=plt.figure(1)
fig.clf()
fig.set_size_inches((6.5,5), forward=True)

ax=fig.add_axes([0,0,1,1])
ax.set_adjustable('datalim')

leg_ax=fig.add_axes([0.0,0.0,0.40,0.25])
leg_bb=leg_ax.get_position()

cax=fig.add_axes([leg_bb.xmin+0.02,
                  leg_bb.ymin+0.10,
                  leg_bb.width-0.04,0.04])

ax.axis('off')

dem_mask=dem.polygon_mask(grid_poly)
dem_clip=dem.copy()
dem_clip.F=dem.F.copy()
dem_clip.F[~dem_mask]=np.nan

#cm_topobathy=scmap.load_gradient('BlueYellowRed.cpt')
cm_topobathy=scmap.transform_color(lambda rgb: 0.8*rgb + 0.2,
                                   cm.inferno)


img=dem_clip.plot(ax=ax,cmap=cm_topobathy)
img.set_clim([-8,4])

ac=aerial.crop(clip).plot(ax=ax,zorder=-2)

grid.plot_edges(lw=0.5,color='k',alpha=0.3,ax=ax)

cbar=plt.colorbar(img,cax=cax,label='Bathymetry (m NAVD88)',
                  orientation='horizontal')
cbar.set_ticks([-8,-5,-2,1,4])

plot_wkb.plot_wkb(grid_poly,edgecolor='k',lw=1.,zorder=3,ax=ax,facecolor='none')

ax.plot(rx_locs['yap_x'], rx_locs['yap_y'], 'yo', label='Hydrophone',
        mew=1, mec='k')
ax.axis(zoom)

leg_ax.legend(loc='upper left',
              frameon=False,
              bbox_to_anchor=[0,0.95],
              handles=ax.lines)

overview_ax=fig.add_axes([0.63,0.55,0.37,0.45])

plot_wkb.plot_wkb(sfe_geom,ax=overview_ax,facecolor='0.5',edgecolor='0.5',lw=0.2)
    
overview_ax.axis('equal')
overview_ax.xaxis.set_visible(0)
overview_ax.yaxis.set_visible(0)

x=0.5*(zoom[0]+zoom[1])
y=0.5*(zoom[2]+zoom[3])
overview_ax.plot( [x], [y], 'r*',ms=7)

# Add release locations in the inset.
overview_ax.plot( [ release_xy[0,0]], [release_xy[0,1]],'bv')
overview_ax.plot( [ release_xy[1,0]], [release_xy[1,1]],'bv')
overview_ax.annotate("Upper\nrelease", release_xy[0,:] + np.r_[-5000,0],
                     ha='right')
overview_ax.annotate("Lower\nrelease", release_xy[1,:] + np.r_[-4000,0],
                     ha='right',va='top')
overview_ax.annotate( "Study\nsite", [x+6000,y])

overview_ax.axis( (527837., 690000., 4131000., 4269000) )

# A few geographic labels
# Better done in inkscape, but try it here...
ax.texts=[]
ax.text(647780, 4185550.,r'$\leftarrow$ San Joaquin R.',
        rotation=18.,fontsize=11)

ax.text(646865, 4185800.,r'$\leftarrow$ Old R.',
        rotation=0.,fontsize=11)

ax.text(647300, 4185950.,r'San Joaquin R. $\rightarrow$',
        rotation=40.,fontsize=11)

plot_utils.scalebar([0.55,0.03],dy=0.02,L=500,ax=ax,
                    fractions=[0,0.2,0.4,1.0],
                    label_txt='m',
                    style='ticks',
                    lw=1.25,
                    xy_transform=ax.transAxes)

fig.savefig('fig_study_area.png',dpi=200)

##

