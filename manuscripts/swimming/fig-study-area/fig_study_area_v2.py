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
from matplotlib import cm,collections
from matplotlib.patches import Rectangle

from shapely import geometry

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

# Legal delta boundary:
delta_bdry=wkb2shp.shp2geom("../../../gis/Legal_Delta_Boundary.shp",
                            target_srs='EPSG:26910')['geom'][0]


##
ca_shp=wkb2shp.shp2geom("../../../gis/CA_State/CA_State_TIGER2016.shp",
                        target_srs='EPSG:26910')

##
from stompy.spatial import wkb2shp

if 0: # Is the cascade grid of any use here?
    cgrid=unstructured_grid.UnstructuredGrid.read_dfm("../../../../cascade/rmk_validation_setups/wy2011/r18b_net.nc")
    cgrid_poly=cgrid.boundary_polygon()
    wkb2shp.wkb2shp('r18b_net_outline.shp',[cgrid_poly],overwrite=True)
if 0:
    sfbo_grid=unstructured_grid.UnstructuredGrid.read_ugrid("../../../../sfb_ocean/suntans/grid-merge-suisun/splice-merge-05-filled-edit70.nc")
    sfbo_poly=sfbo_grid.boundary_polygon()
    wkb2shp.wkb2shp('sfb_ocean_outline.shp',[sfbo_poly],overwrite=True)
if 0:
    csc_grid=unstructured_grid.UnstructuredGrid.read_ugrid("../../../../csc/grid/CacheSloughComplex_v111-edit21.nc")
    csc_poly=csc_grid.boundary_polygon()
    wkb2shp.wkb2shp('csc_outline.shp',[csc_poly],overwrite=True)
if 0:
    sfe_poly=wkb2shp.shp2geom("../../../gis/dem_to_shoreline/shoreline_from_dem.shp")
    # Extend further into ocean
    ca_mpoly=ca_shp['geom'][0].buffer(-5556)
    ocean_poly=ca_mpoly.envelope.difference(ca_mpoly)
    sfe_geom = sfe_poly['geom'][0].union(ocean_poly)

if 0:
    # Extend SJ River
    ca_rivers=wkb2shp.shp2geom("../../../gis/CA_State/hydro/CaliforniaHydro.shp",
                               target_srs='EPSG:26910')
    sel=(ca_rivers['NAME']=='San Joaquin River') # |(ca_rivers['NAME']=='Sacramento River')
    river_geoms=ca_rivers['geom'][sel]
    
    river_polys=[g.buffer(50.0,resolution=4) for g in river_geoms]
    river_poly=ops.cascaded_union(river_polys)

    sfe_geom = sfe_poly['geom'][0].union(ocean_poly)

    # Get rid of small channels
    sfe_geom2=sfe_geom.buffer(-10,resolution=4).buffer(10,resolution=4)

    sfe_geom3=sfe_geom2.union( river_poly )


##

clip_poly=geometry.Polygon(np.array([[ 603340., 4289284.],
                                     [ 644080., 4290464.],
                                     [ 653527., 4227288.],
                                     [ 701942., 4126324.],
                                     [ 691315., 4117468.],
                                     [ 588579., 4204852.],
                                     [ 588579., 4238506.],
                                     [ 585037., 4300502.]]))

##

# Editing all done in QGIS project. Load final result:
shore=wkb2shp.shp2geom('../../../gis/compiled_shoreline.shp')
sfe_geom=shore['geom'][0]
    
##

#zoom=(646818.0, 648074.0, 4185291.632, 4186257.632)
#zoom=(646837., 648093., 4185150., 4186150.)
zoom=(646832.122007244, 648058.6644664974, 4185201.211117178, 4186275.342974986)

zoom2= (644803.2820757322, 655533.962329247,
        4182433.5251442483, 4188541.758519326)

ax_props=dict(fontsize=9)
overview_props=dict(fontsize=9)

fig=plt.figure(1)
fig.clf()
fig.set_size_inches((6.5,3.7), forward=True)

if 0:
    ax=fig.add_axes([0,0,0.65,1])
    overview_ax=fig.add_axes([0.65,0.0,
                              0.35,1.0])
    leg_ax=fig.add_axes([0.0,0.0,0.40,0.25])
    overview_ax2=None
else:
    # overview on left
    ax=fig.add_axes([0.35,0,0.65,1])
    overview_ax2=fig.add_axes([0.35,0,0.28,0.35])
    
    overview_ax=fig.add_axes([0.0,0.0,
                              0.35,1.0])
    #leg_ax=fig.add_axes([0.35,0.0,0.40,0.25])
    leg_ax=fig.add_axes([0.7,0.48,0.30,0.25])



for a in [leg_ax,overview_ax,ax,overview_ax2]:
    if a is None: continue
    plt.setp(list(a.spines.values()),
             visible=0)

#overview_ax2.spines['left'].set_visible(0)
#ax.spines['left'].set_visible(1)
overview_ax.spines['right'].set_visible(1)
overview_ax2.spines['top'].set_visible(1)
overview_ax2.spines['right'].set_visible(1)


ax.set_adjustable('datalim')

leg_ax.xaxis.set_visible(0)
leg_ax.yaxis.set_visible(0)
leg_ax.patch.set_visible(0)


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

grid_coll=grid.plot_edges(lw=0.5,color='k',alpha=0.3,ax=ax)

cbar=plt.colorbar(img,cax=cax,label='Bathymetry (m NAVD88)',
                  orientation='horizontal')

cbar.set_ticks([-8,-5,-2,1,4])
plt.setp(cax.get_xticklabels(),fontsize=9)
plt.setp(cax.xaxis.label,fontsize=9)

plot_wkb.plot_wkb(grid_poly,edgecolor='k',lw=1.,zorder=3,ax=ax,facecolor='none')

ax.plot(rx_locs['yap_x'], rx_locs['yap_y'], 'yo', label='Hydrophone',
        mew=1, mec='k')
ax.axis(zoom)

leg_ax.legend(loc='upper left',
              frameon=False,
              bbox_to_anchor=[0,0.90],
              fontsize=9,
              handles=ax.lines)

plot_wkb.plot_wkb(sfe_geom.intersection(clip_poly),
                  ax=overview_ax,facecolor='#2d74ad',edgecolor='#2d74ad',lw=0.2)
if overview_ax2:
    rect=geometry.box(zoom2[0],zoom2[2],zoom2[1],zoom2[3]).buffer(5000)
    plot_wkb.plot_wkb(sfe_geom.intersection(rect),
                      ax=overview_ax2,facecolor='#2d74ad',edgecolor='#2d74ad',lw=0.2)

for ov_ax in [overview_ax,overview_ax2]:
    ov_ax.axis('equal')
    ov_ax.xaxis.set_visible(0)
    ov_ax.yaxis.set_visible(0)

x=0.5*(zoom[0]+zoom[1])
y=0.5*(zoom[2]+zoom[3])
# overview_ax.plot( [x], [y], 'r*',ms=7)

# Add release locations in the inset.
overview_ax.plot( [ release_xy[0,0]], [release_xy[0,1]],'bv')
overview_ax.plot( [ release_xy[1,0]], [release_xy[1,1]],'bv')
# overview_ax.annotate("Upper\nrelease", release_xy[0,:] + np.r_[-5000,0],
#                      ha='right',**overview_props)
# overview_ax.annotate("Lower\nrelease", release_xy[1,:] + np.r_[-4000,0],
#                      ha='right',va='top',**overview_props)
# overview_ax.annotate( "Study\nsite", [x+6000,y],**overview_props)

# overview_ax.annotate("Chipps\nIsland",
#                      xy=[595218, 4212542.],
#                      xytext=[575000, 4235500.],
#                      arrowprops=dict(arrowstyle="->"),
#                      **overview_props)

# overview_ax.annotate("Pacific\n Ocean",
#                      xy=[532589, 4125054],
#                      **overview_props)

# And gauge locations:
# SJD, HOR, MSD
# 
MSD_ll=[-121.306,37.786]
SJD_ll=[-121.317724,37.822331]
OH1_ll=[-121.331253051758,37.8075523376465]
gauge_ll=np.array([ MSD_ll, SJD_ll, OH1_ll])
gauge_xy=proj_utils.mapper('WGS84','EPSG:26910')(gauge_ll)

ov_ax=overview_ax2 or overview_ax
ov_ax.plot( gauge_xy[:,0], gauge_xy[:,1],'g.')
if not overview_ax2:
    ax.plot( gauge_xy[:,0], gauge_xy[:,1],'g.')

overview_ax.texts=[]
# overview_ax.annotate("Water\nproject\nintakes",
#                      xy=[624818, 4186745.],
#                      xytext=[584000, 4176000.],
#                      arrowprops=dict(arrowstyle="->"),
#                      **overview_props)


overview_ax.axis( (591145.4117244165, 685561.9795310685, 4122628.38358537, 4276184.999358825) )
if overview_ax2:
    overview_ax2.axis(zoom2)
    
    
# A few geographic labels
# Better done in inkscape, but try it here...
ax.texts=[]

sbar=plot_utils.scalebar([0.55,0.03],dy=0.02,L=400,ax=ax,
                         fractions=[0,0.25,0.5,1.0],
                         label_txt='m',
                         style='ticks',
                         lw=1.25,
                         xy_transform=ax.transAxes)

plot_wkb.plot_wkb(delta_bdry,ax=overview_ax,
                  fc='0.85', ec='0.7',ls='-',lw=0.5,zorder=-2)

if 1:
    # Add a location in CA mini-inset
    ca_geom=ca_shp[0]['geom']

    ov_bbox=overview_ax.get_position()

    ca_ax=fig.add_axes( [ov_bbox.xmin+0.62*ov_bbox.width,
                         ov_bbox.ymin+0.73*ov_bbox.height,
                         0.35*ov_bbox.width, 0.25*ov_bbox.height])

    plot_wkb.plot_wkb(# ca_geom,
                      ca_geom.difference(sfe_geom.buffer(-100).buffer(100).simplify(200)),
                      ax=ca_ax,fc='none',ec='k',lw=0.5)
    ca_ax.axis('equal')

    ca_ax.xaxis.set_visible(0)
    ca_ax.yaxis.set_visible(0)
    plt.setp( list(ca_ax.spines.values()), visible=0)

    # show the overview extent on ca_ax.
    xxyy=overview_ax.axis()

    rect=Rectangle([xxyy[0],xxyy[2]], width=xxyy[1]-xxyy[0],
                   height=xxyy[3]-xxyy[2],color='0.75',zorder=-2)

    ca_ax.add_patch(rect)


#r_style=dict(color='r',zorder=-1)
r_style=dict(edgecolor='r',lw=1.5,zorder=-1,facecolor='none')

if 1: # Show the study site as an inset rect:
    # show the overview extent on ca_ax.
    xxyy=ax.axis()

    ov_ax=overview_ax2 or overview_ax
    rect=Rectangle([xxyy[0],xxyy[2]], width=xxyy[1]-xxyy[0],
                   height=xxyy[3]-xxyy[2],**r_style)
    ov_ax.add_patch(rect)

if overview_ax2:
    xxyy=overview_ax2.axis()
    rect=Rectangle([xxyy[0],xxyy[2]], width=xxyy[1]-xxyy[0],
                   height=xxyy[3]-xxyy[2],**r_style)
    overview_ax.add_patch(rect)

sbar2=plot_utils.scalebar([0.15,0.05],dy=0.02,L=40000,
                          ax=overview_ax,
                          fractions=[0,0.25,0.5,1.0],
                          unit_factor=1e-3,
                          label_txt='km',
                          style='ticks',
                          lw=1.25,
                          xy_transform=overview_ax.transAxes)
    
# Drop the texts
# plt.setp(overview_ax.texts,visible=False)
# Change all markers to small dots. 
plt.setp(overview_ax.lines,marker='.',ms=2)

plt.setp(overview_ax.texts,fontsize=9)
plt.setp(ax.texts,fontsize=9)

    

## 
# fig.savefig('fig_study_area.png',dpi=200)
fig.savefig('fig_study_area_v2b.pdf') # seems to import ok.  used 'internal' pdf import

##

# Drop some of the text and save pieces to put this together in
# inkscape

# Background:
plt.setp( ax.texts[:3], visible=False)
overview_ax.set_visible(False)
leg_ax.set_visible(False)
cax.set_visible(False)
#fig.savefig('fig_study_area-ax.svg')
fig.savefig('fig_study_area-ax.pdf') # seems to import ok.  used 'internal' pdf import
#fig.savefig('fig_study_area-ax.png',dpi=200)

## Inset:
plt.setp( ax.texts[:3], visible=False)
overview_ax.set_visible(True)
leg_ax.set_visible(False)
ax.set_visible(False)
cax.set_visible(False)
#fig.savefig('fig_study_area-ax.svg')


extent = overview_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('fig_study_area-overview.pdf',bbox_inches=extent) 

## 

overview_ax.set_visible(False)
leg_ax.set_visible(True)
cax.set_visible(True)

extent = leg_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('fig_study_area-legend.pdf',bbox_inches=extent) 

##


# Additional versions for presentation
grid_coll.set_visible(0)
fig.savefig('fig_study_area-no_grid.png',dpi=200)

##

grid_coll.set_visible(1)
overview_ax.set_visible(0)
ax.lines[0].set_visible(0)
fig.savefig('fig_study_area-no_overview.png',dpi=200)
ax.lines[0].set_visible(1)

##

ax.lines[0].set_visible(0)

# Replace scalebar with smaller one:
lines,texts=sbar
for l in lines:
    if l in ax.collections:
        ax.collections.remove(l)
for t in texts:
    if t in ax.texts:
        ax.texts.remove(t)

ax.axis( (647134.9801316926, 647313.3037316924, 4185782.897096002, 4185920.069096002) )
sbar=plot_utils.scalebar([0.03,0.9],dy=0.02,L=50,ax=ax,
                    fractions=[0,0.25,0.5,1.0],
                    label_txt='m',
                    style='ticks',
                    lw=1.25,
                    xy_transform=ax.transAxes)

cax.set_visible(0)
leg_ax.set_visible(0)
grid_coll.set_alpha(0.8)
fig.savefig('grid-only-zoom.png',dpi=150)

cax.set_visible(1)
leg_ax.set_visible(1)
grid_coll.set_alpha(0.3)

##

# Add in transect locations, zoom in
adcp_transects=wkb2shp.shp2geom("../../../gis/transects_2018.shp")
tran_segs=[ np.array(geom) for geom in adcp_transects['geom']]
tran_lcoll=collections.LineCollection(tran_segs,color='k',lw=1.2)
ax.add_collection(tran_lcoll)
overview_ax.set_visible(0)

# Replace scalebar with smaller one:
lines,texts=sbar
for l in lines:
    if l in ax.collections:
        ax.collections.remove(l)
for t in texts:
    if t in ax.texts:
        ax.texts.remove(t)
        
sbar=plot_utils.scalebar([0.03,0.9],dy=0.02,L=200,ax=ax,
                    fractions=[0,0.25,0.5,1.0],
                    label_txt='m',
                    style='ticks',
                    lw=1.25,
                    xy_transform=ax.transAxes)

ax.axis( (646964., 647585., 4185571., 4186048.) )

fig.savefig('fig_study_area-adcp_transects.png',dpi=200)

##

