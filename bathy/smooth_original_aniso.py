"""
The smoothness of the ADCP bathy seems good, but there are lateral features that we 
don't get as well when using ADCP bathy.

Try a strong anisotropic filter on the original bathy
"""

from stompy.spatial import proj_utils, wkb2shp, field
import matplotlib.pyplot as plt
import xarray as xr
from stompy.spatial import wkb2shp

import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')

##

dem=field.GdalGrid('junction-composite-20190117-no_adcp.tif')
## 

dem_adcp=field.GdalGrid('junction-composite-dem.tif')

##
# just use the computational grid
g=unstructured_grid.UnstructuredGrid.read_ugrid('../model/grid/snubby_junction/snubby-08-edit50.nc')
    
## 
zoom=(647088.821657404, 647702.7390601198, 4185484.2206095303, 4185941.6880934904)

plt.figure(10).clf()

fig,ax=plt.subplots(1,1,num=10)

img1=dem.plot(clim=[-5,2.2],ax=ax,cmap='jet')

g.plot_edges( ax=ax,lw=0.3,color='k')

##

bcs=wkb2shp.shp2geom("../model/grid/snubby_junction/forcing-snubby-01.shp")

## 
dem_crop=dem.crop(g.bounds())

# Use the computational grid to set boundaries, but run the smoothing on the
# DEM so we can mix it in after the fact
gr=unstructured_grid.UnstructuredGrid(max_sides=4)
xmin,xmax,ymin,ymax=dem_crop.extents

# Way overkill..
gr.add_rectilinear( p0=[xmin-dem_crop.dx/2,ymin-dem_crop.dy/2],
                    p1=[xmax+dem_crop.dx/2,ymax+dem_crop.dy/2],
                    nx=dem_crop.F.shape[1]+1,
                    ny=dem_crop.F.shape[0]+1)

##
bounds=g.boundary_polygon()

sel=gr.select_cells_intersecting(bounds)
##
for c in np.nonzero(~sel)[0]:
    gr.delete_cell(c)

gr.delete_orphan_edges()
gr.delete_orphan_nodes()
gr.renumber()
##

zoom=(647088.821657404, 647702.7390601198, 4185484.2206095303, 4185941.6880934904)

plt.figure(10).clf()

fig,ax=plt.subplots(1,1,num=10)

img1=dem.plot(clim=[-5,2.2],ax=ax,cmap='jet')

gr.plot_edges( ax=ax,lw=0.3,color='k')

## 
# Three BCs, six total sets of edges.

edge_sets=[]
for bc in bcs:
    edge_sets.append( gr.select_edges_by_polyline(bc['geom']) )

OR_left =np.array(bcs['geom'][1].coords)[0] # left end at OR
OR_right=np.array(bcs['geom'][1].coords)[-1] # right end at OR
SJ_up_left=np.array(bcs['geom'][0].coords)[0]
SJ_up_right=np.array(bcs['geom'][0].coords)[-1]
SJ_dn_right=np.array(bcs['geom'][2].coords)[-1]
SJ_dn_left=np.array(bcs['geom'][2].coords)[0]

river_left = gr.select_nodes_boundary_segment( coords=[OR_left,SJ_up_left] ) 
river_right= gr.select_nodes_boundary_segment( coords=[SJ_up_right,SJ_dn_right] )

river_split= gr.select_nodes_boundary_segment( coords=[SJ_dn_left,OR_right] )


# cells:
river_left_cells =np.unique( [ c for n in river_left for c in gr.node_to_cells(n) ] )
river_right_cells=np.unique( [ c for n in river_right for c in gr.node_to_cells(n) ] )
river_split_cells=np.unique( [ c for n in river_split for c in gr.node_to_cells(n) ] )

##
plt.figure(12).clf()
gr.plot_edges(lw=0.5,color='k')
gr.plot_cells(mask=river_left_cells,color='r',alpha=0.5)
## 

from stompy.model import unstructured_diffuser

diff=unstructured_diffuser.Diffuser(gr)

for c in river_left_cells:
    diff.set_dirichlet(0,cell=c)

for c in river_right_cells:
    diff.set_dirichlet(100,cell=c)

for c in river_split_cells:
    diff.set_dirichlet(50,cell=c)
    
diff.construct_linear_system()
diff.solve_linear_system(animate=False)

psi=diff.C_solved

##

# Can get better interpolation on the dual
gd=gr.create_dual(create_cells=True)

## 

psi_node=psi[ gd.nodes['dual_cell'] ]

##

plt.figure(10).clf()
fig,ax=plt.subplots(1,1,num=10)

img1=dem_crop.plot(clim=[-5,2.2],ax=ax,cmap='gray')

gd.contour_node_values(psi_node,40,linewidths=0.5,colors='orange')

ax.axis(zoom)

##

cc=gr.cells_center()
d_orig=dem(cc)

d_med=d_orig.copy()

for c in utils.progress(gr.valid_cell_iter()):
    # Get a large-ish neighborhood:
    # Would like to make this a length scale to be more grid invariant
    # 45 seemed to fill in troughs too much
    L=35
    nbrs=np.nonzero( utils.mag( cc-cc[c])<L)[0]

    # what does this typically look like in terms of radius?
    dist2=utils.mag( cc[nbrs]-cc[c] )
    # print('Max dist: ',dist2.max())
    
    alpha=10 # controls the degree of anistropy
    coords=np.c_[ cc[nbrs], alpha*psi[nbrs]]
    coord0=np.array( [ cc[c,0],cc[c,1],alpha*psi[c] ] )

    dists=utils.mag(coords-coord0)

    # Will take the median of a subset of those:
    N=int(len(dists)*0.15)
    subsel=np.argsort(dists)[:N]
    close_nbrs=nbrs[subsel]
    d_med[c] = np.median( d_orig[close_nbrs] )


plt.figure(13).clf()
fig,axs=plt.subplots(1,3,num=13,sharex=True,sharey=True)
axs[0].set_adjustable('box',share=True)

clim=[-4,0]
gr.plot_cells(values=d_orig,cmap=turbo,clim=clim,ax=axs[0])
gr.plot_cells(values=d_med ,cmap=turbo,clim=clim,ax=axs[1])
gr.plot_cells(values=d_med-d_orig,cmap='coolwarm',clim=[-1,1],ax=axs[2])

axs[0].axis(zoom)
fig.tight_layout()
##
from stompy.spatial import field
dem_smooth=field.XYZField(gr.cells_center(),d_med).rectify()

dem_smooth.write_gdal('dwr_smooth_v01.tif')
