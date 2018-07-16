"""
Apply the interpolation method of Zhang 2016 to this model domain.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from stompy import utils
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy.spatial import wkb2shp, proj_utils
import stompy.plot.cmap as scmap
cmap=scmap.load_gradient('hot_desaturated.cpt')

##

# This is written out by merge_maps.py, currently just for one timestamp.
ds=xr.open_dataset('merged_map.nc')
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

##

zoom=(646650, 648263, 4185320., 4186325.)

fig=plt.figure(1)
fig.clf()
fig,ax=plt.subplots(num=1)

# g.plot_edges(ax=ax,lw=0.4,color='k')
ccoll=g.plot_cells(values=ds.ucxa.values,ax=ax,cmap=cmap,clip=zoom)
ccoll.set_clim([-1,1])
ax.axis('equal')

##

# clip=(646650, 648263, 4185320., 4186325.)
# This one is tighter around the ADCP samples
clip=(646966, 647602, 4185504, 4186080)

# Build a pseudo grid that has edges for neighbors (like a dual, but including
# a larger neighborhood.
# For now, this is starting from the original grid, but it could be
# done on a cartesian DEM grid, too.

# transit time network
t_net=unstructured_grid.UnstructuredGrid(extra_node_fields=[('value',np.float64),
                                                            ('cell',np.int32)],
                                         extra_edge_fields=[('Td',np.float64)])

gc_to_t_net_node={} # speed up the reverse lookup
cc=g.cells_center()
for c in np.nonzero(g.cell_clip_mask(clip))[0]:
    n=t_net.add_node(x=cc[c],value=np.nan,cell=c)
    gc_to_t_net_node[c]=n

# Create the neighborhood connectivity
for n in range(t_net.Nnodes()):
    c=t_net.nodes['cell'][n]
    # use nodes to get the neighborhood of cells
    c_nodes=g.cell_to_nodes(c)
    c_cells=[]
    for c_node in c_nodes:
        c_cells+=list(g.node_to_cells(c_node))
    c_cells=set(c_cells)
    for cnbr in c_cells:
        if (cnbr==c) or (cnbr not in gc_to_t_net_node):
            # if we clipped the original, this will have a few
            # hits outside the clip, which we ignore.
            continue
        try:
            t_net.add_edge(nodes=[n,gc_to_t_net_node[cnbr]])
        except t_net.GridException:
            pass # b/c the edge is already there - great.

##
if 0:
    # It looks a little crazy around the triangles, crazy, but
    # I think correct
    zoom=(646650, 648263, 4185320., 4186325.)

    fig=plt.figure(2)
    fig.clf()
    fig,ax=plt.subplots(num=2)

    t_net.plot_edges(ax=ax,lw=0.3,color='k')
    ax.axis('equal')

##

t_net.add_node_field( 'u',ds.ucxa.values[t_net.nodes['cell']],on_exists='overwrite')
t_net.add_node_field( 'v',ds.ucya.values[t_net.nodes['cell']],on_exists='overwrite')
eps=1e-5
for j in range(t_net.Nedges()):
    if j%10000==0:
        print("%d/%d"%(j,t_net.Nedges()))
    n1,n2=nodes=t_net.edges['nodes'][j]
    xy=t_net.nodes['x'][nodes]
    dx,dy=xy[1,:] - xy[0,:]
    u=t_net.nodes['u'][nodes].sum()
    v=t_net.nodes['v'][nodes].sum()
    # can't drop signs until here, otherwise we lose quadrant information
    # doing it here also allows for protecting against division by zero.
    Tab=(2*(dx**2 +dy**2))/(eps+abs(u*dx+v*dy))
    t_net.edges['Td'][j]=Tab

##

if 0:
    zoom=(646650, 648263, 4185320., 4186325.)

    fig=plt.figure(2)
    fig.clf()
    fig,ax=plt.subplots(num=2)

    ecoll=t_net.plot_edges(values=t_net.edges['Td'],ax=ax,lw=1.,cmap=cmap)
    ecoll.set_clim([2,50])
    ax.axis('equal')

##

import bathy
dem=bathy.dem()

##

adcp_shp=wkb2shp.shp2geom('derived/samples-depth.shp')
adcp_ll=np.array( [ np.array(pnt) for pnt in adcp_shp['geom']] )
adcp_xy=proj_utils.mapper('WGS84','EPSG:26910')(adcp_ll)
adcp_xyz=np.c_[ adcp_xy,adcp_shp['depth'] ]

##

# Rather than use the ADCP data directly, during testing
# use its horizontal distribution, but pull "truth" from the
# DEM
true_xyz=adcp_xyz.copy()
true_xyz[:,2] = dem( true_xyz[:,:2] )

##
if 0:
    plt.figure(10).clf()
    plt.plot( true_xyz[:,2], adcp_xyz[:,2], 'k.',ms=1,alpha=0.3)
    plt.xlabel('DEM (m)')
    plt.ylabel('ADCP (m)')
    plt.grid(1)
    plt.plot( [-10,4],[-10,4],'g-',zorder=-1)

##

xyz_input=true_xyz

# Put those samples onto nodes:
xyz_to_cell=np.zeros(len(xyz_input),np.int32)-1

for i,xyz in enumerate(xyz_input):
    if i%1000==0:
        print("%d/%d"%(i,len(xyz_input)))
    # select by cell in the original grid to be sure that
    # samples fall within that "control volume"
    c=g.select_cells_nearest(xyz[:2],inside=True)
    if c is not None:
        xyz_to_cell[i]=c

##

t_net.nodes['value'][:]=np.nan # make sure we're starting clean

# 18k points, but they map onto 2k cells.
# Install the averages onto the values field of the t_net nodes
for c,hits in utils.enumerate_groups(xyz_to_cell):
    if c not in gc_to_t_net_node:
        continue
    n=gc_to_t_net_node[c]
    t_net.nodes['value'][n] = xyz_input[:,2][hits].mean()

##

if 0:
    plt.figure(11).clf()
    t_net.plot_edges(lw=0.4,color='k')
    t_net.plot_nodes(values=t_net.nodes['value'],zorder=2)

##


# 8k nodes
# 2k samples
# That would be 16M Dijkstra calls.
# If instead we use breadth first?
search_n=8 # that's what they concluded was best in the paper

new_values=t_net.nodes['value'].copy()
samples=np.nonzero( np.isfinite(t_net.nodes['value']) )[0]

for n in range(t_net.Nnodes()):
    if n%1000==0:
        print("%d/%d"%(n,t_net.Nnodes()))

    if np.isfinite(t_net.nodes['value'][n]):
        continue # this is a sample
    elif (t_net.nodes['u'][n]==0) and (t_net.nodes['v'][n]==0):
        continue # not wet in the model

    # search from n along edges of t_net, with cost Td,
    # finding the search_n nearest samples.n
    results=t_net.shortest_path(n,samples,edge_weight=lambda j:t_net.edges['Td'][j],
                                return_type='cost',max_return=search_n)
    r_nodes,r_costs = zip(*results)
    r_nodes=list(r_nodes) 
    result_values=t_net.nodes['value'][r_nodes]
    assert np.all( np.isfinite(result_values) )
    weights=1./np.array(r_costs)
    weights /= weights.sum()

    new_values[n]=(weights*result_values).sum()

# - original
# In [110]: np.argsort(sample_dists)[:10]
# Out[110]: array([50, 60, 51, 72, 73, 61, 52, 85, 71, 97])
# after updating code for multi-point, but still using single point search:
#           array([50, 60, 51, 72, 73, 61, 52, 85, 71, 97])
# as multi-target breadth-first:
# In [130]: np.searchsorted(samples,[r[0] for r in results])
# Out[130]: array([50, 60, 51, 72, 73, 61, 52, 85, 71, 97])


##

fig=plt.figure(10)
fig.clf()
fig,ax=plt.subplots(num=10)

t_net.plot_edges(ax=ax,color='k',lw=0.5,alpha=0.3)
t_net.plot_nodes(ax=ax,values=new_values,sizes=40,cmap=cmap)

##

