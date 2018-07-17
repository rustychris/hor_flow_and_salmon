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

import bathy
dem=bathy.dem()

##

class InterpZhang(object):
    def __init__(self,g,u,v,clip=None):
        """
        g: UnstructuredGrid on which velocity is specified
        u: cell-centered eastward velocity
        v: cell-centered northward velocity
        clip: (x,x,y,y) bounding box to analyze subset of g.
        """
        if clip is None:
            clip=g.bounds()
        self.clip=clip

        self.g=g
        self.uv=np.c_[u,v]

        self.build_dual()
        self.apply_velocity_field()

    def build_dual(self):
        """
        Build a pseudo grid that has edges for neighbors (like a dual, but including
        a larger neighborhood.
        For now, this is starting from the original grid, but it could be
        done on a cartesian DEM grid, too.
        """

        # transit time network
        self.t_net=unstructured_grid.UnstructuredGrid(extra_node_fields=[('value',np.float64),
                                                                         ('cell',np.int32)],
                                                      extra_edge_fields=[('Td',np.float64)])

        self.gc_to_t_net_node={} # speed up the reverse lookup
        cc=g.cells_center()
        for c in np.nonzero(self.g.cell_clip_mask(clip))[0]:
            n=self.t_net.add_node(x=cc[c],value=np.nan,cell=c)
            self.gc_to_t_net_node[c]=n

        # Create the neighborhood connectivity
        for n in range(self.t_net.Nnodes()):
            c=self.t_net.nodes['cell'][n]
            # use nodes to get the neighborhood of cells
            c_nodes=self.g.cell_to_nodes(c)
            c_cells=[]
            for c_node in c_nodes:
                c_cells+=list(self.g.node_to_cells(c_node))
            c_cells=set(c_cells)
            for cnbr in c_cells:
                if (cnbr==c) or (cnbr not in self.gc_to_t_net_node):
                    # if we clipped the original, this will have a few
                    # hits outside the clip, which we ignore.
                    continue
                try:
                    self.t_net.add_edge(nodes=[n,self.gc_to_t_net_node[cnbr]])
                except self.t_net.GridException:
                    pass # b/c the edge is already there - great.

    def apply_velocity_field(self):
        """
        Assign per-node velocities and per-edge travel times.
        """
        cells=self.t_net.nodes['cell']
        self.t_net.add_node_field( 'u',self.uv[cells,0],on_exists='overwrite')
        self.t_net.add_node_field( 'v',self.uv[cells,1],on_exists='overwrite')
        eps=1e-5
        for j in range(self.t_net.Nedges()):
            if j%10000==0:
                print("%d/%d"%(j,self.t_net.Nedges()))
            n1,n2=nodes=self.t_net.edges['nodes'][j]
            xy=self.t_net.nodes['x'][nodes]
            dx,dy=xy[1,:] - xy[0,:]
            u=self.t_net.nodes['u'][nodes].sum()
            v=self.t_net.nodes['v'][nodes].sum()
            # can't drop signs until here, otherwise we lose quadrant information
            # doing it here also allows for protecting against division by zero.
            Tab=(2*(dx**2 +dy**2))/(eps+abs(u*dx+v*dy))
            self.t_net.edges['Td'][j]=Tab
    def set_samples(self,xyz_input):
        """
        Set the observed data as [N,3] array, x,y, and depth
        """
        # Get the mapping of sample location to cell, with
        # potentially multiple samples mapping to the same cell.
        xyz_to_cell=np.zeros(len(xyz_input),np.int32)-1

        for i,xyz in enumerate(xyz_input):
            if i%1000==0:
                print("%d/%d"%(i,len(xyz_input)))
            # select by cell in the original grid to be sure that
            # samples fall within that "control volume"
            c=self.g.select_cells_nearest(xyz[:2],inside=True)
            if c is not None:
                xyz_to_cell[i]=c

        # make sure we're starting clean
        self.t_net.nodes['value'][:]=np.nan
        # Install the averages onto the values field of the t_net nodes
        for c,hits in utils.enumerate_groups(xyz_to_cell):
            if c not in self.gc_to_t_net_node:
                continue # not part of the clipped region
            n=self.gc_to_t_net_node[c]
            # This could be smarter about locally fitting a surface
            # and get a proper circumcenter value.
            self.t_net.nodes['value'][n] = xyz_input[:,2][hits].mean()

    # The number of nearest samples to consider
    # 8 is what was reported in the Zhang paper
    search_n=8
    def interpolate(self):
        """
        Set self.t_net.nodes['int_value'] by interpolation.
        Dry points are left nan.
        """
        new_values=self.t_net.nodes['value'].copy()
        samples=np.nonzero( np.isfinite(self.t_net.nodes['value']) )[0]

        for n in range(self.t_net.Nnodes()):
            if n%1000==0:
                print("%d/%d"%(n,self.t_net.Nnodes()))

            if np.isfinite(self.t_net.nodes['value'][n]):
                continue # this is a sample
            elif (self.t_net.nodes['u'][n]==0) and (t_net.nodes['v'][n]==0):
                continue # not wet in the model, or still water.

            # search from n along edges of t_net, with cost Td,
            # finding the search_n nearest samples.n
            results=self.t_net.shortest_path(n,samples,edge_weight=lambda j:self.t_net.edges['Td'][j],
                                             return_type='cost',max_return=self.search_n)
            r_nodes,r_costs = zip(*results) # "transpose" results
            r_nodes=list(r_nodes) # tuple causes problems, make it a list.
            result_values=self.t_net.nodes['value'][r_nodes]
            assert np.all( np.isfinite(result_values) )
            weights=1./np.array(r_costs)
            weights/=weights.sum()
            new_values[n]=(weights*result_values).sum()

        self.t_net.add_node_field('int_value',new_values,on_exists='overwrite')
##

adcp_shp=wkb2shp.shp2geom('derived/samples-depth.shp')
adcp_ll=np.array( [ np.array(pnt) for pnt in adcp_shp['geom']] )
adcp_xy=proj_utils.mapper('WGS84','EPSG:26910')(adcp_ll)
adcp_xyz=np.c_[ adcp_xy,adcp_shp['depth'] ]

##

# Rather than use the ADCP data directly, during testing
# use its horizontal distribution, but pull "truth" from the
# DEM
xyz_input=adcp_xyz.copy()
xyz_input[:,2] = dem( xyz_input[:,:2] )

##

# This is written out by merge_maps.py, currently just for one timestamp.
ds=xr.open_dataset('merged_map.nc')
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

iz=InterpZhang(g,
               u=ds.ucxa.values,
               v=ds.ucya.values,
               clip=(646966, 647602, 4185504, 4186080))
iz.set_samples(xyz_input)
iz.interpolate()

##

if 0:
    plt.figure(11).clf()
    t_net.plot_edges(lw=0.4,color='k')
    t_net.plot_nodes(values=t_net.nodes['value'],zorder=2)

##

fig=plt.figure(10)
fig.clf()
fig,ax=plt.subplots(num=10)

t_net=iz.t_net
t_net.plot_edges(ax=ax,color='k',lw=0.5,alpha=0.3)
t_net.plot_nodes(ax=ax,values=t_net.nodes['int_value'],sizes=40,cmap=cmap)

##

cell_values=np.zeros( iz.g.Ncells(),'f8' )
cell_values[:]=np.nan
cell_values[iz.t_net.nodes['cell']]=iz.t_net.nodes['int_value']

plt.figure(11).clf()
fig.clf()
fig,ax=plt.subplots(num=11)

iz.g.plot_edges(color='k',lw=0.2,clip=clip)
ccoll=iz.g.plot_cells(values=cell_values,
                      mask=np.isfinite(cell_values),clip=clip,cmap=cmap,ax=ax)
ccoll.set_clim([-8,3])
plot_utils.cbar(ccoll)

ax.plot(xyz_input[:,0],xyz_input[:,1],'k.',ms=0.3)

ax.axis('equal')

ax.axis( (646917., 647721, 4185492., 4186106.) )
# fig.savefig('zhang-on-grid.png')

##

cc=iz.g.cells_center()
# Grab point-wise samples from original DEM for comparison
dem_values=dem(cc)

##

plt.figure(12).clf()
fig.clf()
fig,ax=plt.subplots(num=12)

iz.g.plot_edges(color='k',lw=0.2,clip=clip)
ccoll=iz.g.plot_cells(values=cell_values-dem_values,
                      mask=np.isfinite(cell_values),clip=clip,cmap='seismic',ax=ax)
ccoll.set_clim([-1,1])
plot_utils.cbar(ccoll,label='Interp-DEM (m)')

ax.plot(xyz_input[:,0],xyz_input[:,1],'k.',ms=0.3)
ax.axis('equal')

ax.axis( (646917., 647721, 4185492., 4186106.) )
fig.tight_layout()
# fig.savefig('zhang-on-grid-errors.png')
