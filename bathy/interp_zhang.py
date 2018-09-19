"""
Apply the interpolation method of Zhang 2016 to this model domain.
"""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from stompy import utils
from stompy.grid import unstructured_grid
from stompy.plot import plot_utils
from stompy.model.delft import dfm_grid
from stompy.spatial import wkb2shp, proj_utils
import stompy.plot.cmap as scmap
cmap=scmap.load_gradient('hot_desaturated.cpt')
six.moves.reload_module(unstructured_grid)

from sklearn.cluster import AgglomerativeClustering

##

import bathy
dem=bathy.dem()

##

class InterpZhang(object):
    idw_L=2.0 # length scale for inverse distance weighting
    idw_k=2.  # inverse power for inverse distance weighting

    def __init__(self,g,u,v,clip=None,**kwargs):
        """
        g: UnstructuredGrid on which velocity is specified
        u: cell-centered eastward velocity
        v: cell-centered northward velocity
        clip: (x,x,y,y) bounding box to analyze subset of g.
        """
        self.__dict__.update(kwargs)

        if clip is None:
            clip=g.bounds()

        # convoluted, but this helps round trip the right raster
        # coordinates
        self.clip=[clip[0]-self.dx/2,clip[1]+self.dx/2,
                   clip[2]-self.dy/2,clip[3]+self.dy/2]

        self.g=g
        self.uv=np.c_[u,v]

        self.build_net()
        self.set_velocity_on_net()
        self.t_net.edges['Td'] = self.calc_Td()

    def calc_Td(self):
        Ne=self.t_net.Nedges()
        Td=np.zeros(Ne,np.float64)

        eps=1e-5
        for j in range(Ne):
            if j%10000==0:
                print("%d/%d"%(j,Ne))
            n1,n2=nodes=self.t_net.edges['nodes'][j]
            xy=self.t_net.nodes['x'][nodes]
            dx,dy=xy[1,:] - xy[0,:]
            u=self.t_net.nodes['u'][nodes].sum()
            v=self.t_net.nodes['v'][nodes].sum()

            Td[j]=self.calc_Td_single(j,u,v,dx,dy)
        return Td
    def calc_Td_single(self,j,u,v,dx,dy):
        # can't drop signs until here, otherwise we lose quadrant information
        # doing it here also allows for protecting against division by zero.
        u_tan=abs(u*dx+v*dy)
        if u_tan!=0.0:
            Tab=(2*(dx**2 +dy**2))/u_tan
        else:
            Tab=np.nan
        return Tab

    def set_samples(self,xyz_input):
        """
        Set the observed data as [N,3] array, x,y, and depth
        """
        # Get the mapping of sample location to dual cell, with
        # potentially multiple samples mapping to the same cell.
        xyz_to_cell=np.zeros(len(xyz_input),np.int32)-1

        for i,xyz in enumerate(xyz_input):
            if i%1000==0:
                print("%d/%d"%(i,len(xyz_input)))
            # select by cell in the original grid to be sure that
            # samples fall within that "control volume"
            c=self.t_dual.select_cells_nearest(xyz[:2],inside=True)
            if c is not None:
                xyz_to_cell[i]=c

        # make sure we're starting clean
        self.t_net.nodes['value'][:]=np.nan
        # Install the averages onto the values field of the t_net nodes
        for c,hits in utils.enumerate_groups(xyz_to_cell):
            if c not in self.dual_to_net_node:
                continue # not part of the clipped region
            n=self.dual_to_net_node[c]
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
            elif (self.t_net.nodes['u'][n]==0) and (self.t_net.nodes['v'][n]==0):
                continue # not wet in the model, or still water.

            new_values[n] = self.interpolate_single(n,samples)

        self.t_net.add_node_field('int_value',new_values,on_exists='overwrite')
    def interpolate_single(self,n,samples):
        """
        compute interpolated value for node n, based on self.t_net_nodes['value']
        for nodes in the set samples (bool mask).
        """
        results=self.neighbors_single(n,samples)
        if len(results)==0:
            return np.nan

        r_nodes,r_costs = zip(*results) # "transpose" results
        r_nodes=list(r_nodes) # tuple causes problems, make it a list.
        result_values=self.t_net.nodes['value'][r_nodes]
        assert np.all( np.isfinite(result_values) )

        # weights=(self.idw_L+np.array(r_costs))**(-self.idw_k)
        # follow Ed's advice on hard clipping distance.
        weights=(np.array(r_costs).clip(self.idw_L,np.inf))**(self.idw_k)
        weights/=weights.sum()
        return (weights*result_values).sum()

    def neighbors_single(self,n,samples):
        """
        Find the neighbors for node n, amongst bitmask samples.
        """
        return self.t_net.shortest_path(n,samples,edge_weight=lambda j:self.t_net.edges['Td'][j],
                                        return_type='cost',max_return=self.search_n)

class InterpZhangDual(InterpZhang):
    def build_net(self):
        """
        Build a pseudo grid that has edges for neighbors (like a dual, but including
        a larger neighborhood.
        """
        self.t_dual=self.g

        # transit time network
        self.t_net=unstructured_grid.UnstructuredGrid(extra_node_fields=[('value',np.float64),
                                                                         ('dual_cell',np.int32),
                                                                         ('hydro_cell',np.int32)],
                                                      extra_edge_fields=[('Td',np.float64)])

        self.dual_to_net_node={} # speed up the reverse lookup
        cc=g.cells_center()
        for c in np.nonzero(self.g.cell_clip_mask(self.clip))[0]:
            n=self.t_net.add_node(x=cc[c],value=np.nan,dual_cell=c,hydro_cell=c)
            self.dual_to_net_node[c]=n

        # Create the neighborhood connectivity
        for n in range(self.t_net.Nnodes()):
            c=self.t_net.nodes['dual_cell'][n]
            # use nodes to get the neighborhood of cells
            c_nodes=self.g.cell_to_nodes(c)
            c_cells=[]
            for c_node in c_nodes:
                c_cells+=list(self.g.node_to_cells(c_node))
            c_cells=set(c_cells)
            for cnbr in c_cells:
                if (cnbr==c) or (cnbr not in self.dual_to_net_node):
                    # if we clipped the original, this will have a few
                    # hits outside the clip, which we ignore.
                    continue
                try:
                    self.t_net.add_edge(nodes=[n,self.dual_to_net_node[cnbr]])
                except self.t_net.GridException:
                    pass # b/c the edge is already there - great.

    def set_velocity_on_net(self):
        """
        Assign per-node velocities
        """
        cells=self.t_net.nodes['hydro_cell']
        self.t_net.add_node_field( 'u',self.uv[cells,0],on_exists='overwrite')
        self.t_net.add_node_field( 'v',self.uv[cells,1],on_exists='overwrite')

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
            if c not in self.dual_to_net_node:
                continue # not part of the clipped region
            n=self.dual_to_net_node[c]
            # This could be smarter about locally fitting a surface
            # and get a proper circumcenter value.
            self.t_net.nodes['value'][n] = xyz_input[:,2][hits].mean()


class InterpZhangCartesian(InterpZhang):
    dx=5.0
    dy=5.0

    def build_net(self):
        """
        Build a cartesian grid that has edges for neighbors
        """
        bounds=self.clip
        quant=lambda a,da: da*(a//da)

        p0=[ quant(bounds[0],self.dx),
             quant(bounds[2],self.dy) ]
        p1=[ quant(bounds[1],self.dx),
             quant(bounds[3],self.dy) ]
        nx=int(round((p1[0]-p0[0])/self.dx))
        ny=int(round((p1[1]-p0[1])/self.dy))

        # cartesian dual, with each cell being a pixel of the end DEM
        self.t_dual=unstructured_grid.UnstructuredGrid()
        # nx,ny are number of intervals, but add_rectilinear wants
        # number of nodes, so add 1 to avoid fencepost error
        self.cart_map = self.t_dual.add_rectilinear( p0,p1,nx+1,ny+1 )
        # cart_map gets ['cells'] => [nx,ny] array to cell index
        # similar for nodes.
        dual_cell_to_g=np.zeros( self.t_dual.Ncells(), np.int32)-1
        # clip to the region of the hydro grid.
        to_delete=[]
        for c,cc in enumerate(self.t_dual.cells_center()):
            g_c=self.g.select_cells_nearest(cc,inside=True)
            if g_c is None:
                to_delete.append(c)
            else:
                dual_cell_to_g[c]=g_c
        self.t_dual.add_cell_field('hydro_cell',dual_cell_to_g)
        for c in to_delete:
            self.t_dual.delete_cell(c)

        # need to toss unused nodes, edges, too.
        self.renumber_maps=self.t_dual.renumber()

        # transit time network
        self.t_net=unstructured_grid.UnstructuredGrid(extra_node_fields=[('value',np.float64),
                                                                         ('dual_cell',np.int32),
                                                                         ('hydro_cell',np.int32)],
                                                      extra_edge_fields=[('Td',np.float64)])

        self.dual_to_net_node={} # speed up the reverse lookup
        cc=self.t_dual.cells_center()
        for c in np.nonzero(self.t_dual.cell_clip_mask(self.clip))[0]:
            n=self.t_net.add_node(x=cc[c],value=np.nan,
                                  dual_cell=c,
                                  hydro_cell=self.t_dual.cells['hydro_cell'][c])
            self.dual_to_net_node[c]=n

        # Create the neighborhood connectivity
        for n in range(self.t_net.Nnodes()):
            c=self.t_net.nodes['dual_cell'][n]
            # use nodes to get the neighborhood of cells
            c_nodes=self.t_dual.cell_to_nodes(c)
            c_cells=[]
            for c_node in c_nodes:
                c_cells+=list(self.t_dual.node_to_cells(c_node))
            c_cells=set(c_cells)
            for cnbr in c_cells:
                if (cnbr==c) or (cnbr not in self.dual_to_net_node):
                    # if we clipped the original, this will have a few
                    # hits outside the clip, which we ignore.
                    continue
                try:
                    self.t_net.add_edge(nodes=[n,self.dual_to_net_node[cnbr]])
                except self.t_net.GridException:
                    pass # b/c the edge is already there - great.

    def set_velocity_on_net(self):
        """
        Assign per-node velocities and per-edge travel times.
        """
        hydro_cells=self.t_net.nodes['hydro_cell']
        self.t_net.add_node_field( 'u',self.uv[hydro_cells,0],on_exists='overwrite')
        self.t_net.add_node_field( 'v',self.uv[hydro_cells,1],on_exists='overwrite')

    def as_raster(self):
        # Have the values on t_dual cells, which are a subset of the
        # the dense grid
        raster_values=np.nan*np.zeros( self.cart_map['cells'].shape,np.float64)

        ren_cell_map=self.renumber_maps['cell_map']
        raster_targets=ren_cell_map[self.cart_map['cells']]
        valid=raster_targets>=0

        # This is a little roundabout given that we know
        # the t_net nodes are equivalent to t_dual cells.  but this way
        # is more general, and not that slow.
        cell_values=np.zeros( self.t_dual.Ncells(),'f8' )
        cell_values[:]=np.nan
        cell_values[self.t_net.nodes['dual_cell']]=self.t_net.nodes['int_value']

        raster_values[valid]=cell_values[raster_targets[valid]]
        # cart_map is organized x,y, so transpose to get the row,col
        # order that SimpleGrid expects
        raster_values=raster_values.T

        # also note that SimpleGrid expects extents based on centers,
        # while clip gave the node indices.  so offset a half pixel here
        f=field.SimpleGrid( extents=[self.clip[0]+self.dx/2,
                                     self.clip[1]-self.dx/2,
                                     self.clip[2]+self.dy/2,
                                     self.clip[3]-self.dy/2],
                            F=raster_values)
        return f

class InterpZhangCartesianBidir(InterpZhangCartesian):
    """
    Td is calculated as a signed quantity, with positive meaning
    that flow is going edges['nodes'][0] to edges['nodes'][1],
    and negative the opposite.
    """
    def __init__(self,*a,**kw):
        super(InterpZhangCartesianBidir,self).__init__(*a,**kw)

    def calc_Td_single(self,j,u,v,dx,dy):
        # For bidirectional, keep the sign here.
        # dx and dy are signed and represent the natural direction
        # of the edge
        u_tan=u*dx+v*dy

        if u_tan!=0.0:
            Tab=(2*(dx**2 +dy**2))/u_tan
        else:
            Tab=np.nan
        return Tab

    def neighbors_single(self,n,samples):
        """
        Return list of neighboring samples for node n, as list
        of [(n,cost),...]
        """
        # search from n along edges of t_net, with cost Td,
        # finding the search_n nearest samples.n
        def fwd_weight(j,dir):
            Td=dir*self.t_net.edges['Td'][j]
            if Td>=0:
                return Td
            else:
                return np.nan

        def rev_weight(j,dir):
            Td=-dir*self.t_net.edges['Td'][j]
            if Td>=0:
                return Td
            else:
                return np.nan

        results_fwd=self.t_net.shortest_path(n,samples,edge_weight=fwd_weight,directed=True,
                                             return_type='cost',max_return=self.search_n//2)
        results_rev=self.t_net.shortest_path(n,samples,edge_weight=rev_weight,directed=True,
                                             return_type='cost',max_return=self.search_n//2)

        # results are lists of (n,cost)
        return results_fwd+results_rev  # concatenate

##

adcp_shp=wkb2shp.shp2geom('derived/samples-depth.shp')
adcp_ll=np.array( [ np.array(pnt) for pnt in adcp_shp['geom']] )
adcp_xy=proj_utils.mapper('WGS84','EPSG:26910')(adcp_ll)
adcp_xyz=np.c_[ adcp_xy,adcp_shp['depth'] ]

##

src='dem'
cluster=True

xyz_input=adcp_xyz.copy()
if src=='dem':
    # Rather than use the ADCP data directly, during testing
    # use its horizontal distribution, but pull "truth" from the
    # DEM
    xyz_input[:,2] = dem( xyz_input[:,:2] )

if cluster:
    linkage='complete'
    n_clusters=3000
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(xyz_input[:,:2])

    group_xyz=np.zeros( (n_clusters,3) )
    for grp,members in utils.enumerate_groups(clustering.labels_):
        group_xyz[grp] = xyz_input[members].mean(axis=0)
    xyz_input=group_xyz

##

# This is written out by merge_maps.py, currently just for one timestamp.
ds=xr.open_dataset('merged_map.nc')
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

##

if 0:
    iz=InterpZhangDual(g,
                       u=ds.ucxa.values,
                       v=ds.ucya.values,
                       clip=(646966, 647602, 4185504, 4186080))
    iz.set_samples(xyz_input)
    iz.interpolate()
if 0:
    iz=InterpZhangCartesian(g,
                            u=ds.ucxa.values,
                            v=ds.ucya.values,
                            clip=(646966, 647602, 4185504, 4186080),
                            dx=2,dy=2)

    iz_search_n=40 # 8 was not enough. 20 wasn't enough either.
    iz.set_samples(xyz_input)
    iz.interpolate()
if 1:
    iz=InterpZhangCartesianBidir(g,
                                 u=ds.ucxa.values,
                                 v=ds.ucya.values,
                                 clip=(646966, 647602, 4185504, 4186080),
                                 dx=2,dy=2)
    # this really shouldn't be too high -- high numbers behave poorly
    # near boundaries, since there aren't enough good samples in
    # one of the directions
    iz.idw_L=3
    iz.search_n=10
    iz.set_samples(xyz_input)
    iz.interpolate()

##

cell_values=np.zeros( iz.t_dual.Ncells(),'f8' )
cell_values[:]=np.nan

cell_values[iz.t_net.nodes['dual_cell']]=iz.t_net.nodes['int_value']

plt.figure(11).clf()
fig,ax=plt.subplots(num=11)
ccoll=iz.t_dual.plot_cells(values=cell_values,
                           mask=np.isfinite(cell_values),clip=clip,cmap=cmap,ax=ax)
ccoll.set_clim([-8,3])
plot_utils.cbar(ccoll)

ax.plot(xyz_input[:,0],xyz_input[:,1],'k.',ms=0.3)

ax.axis('equal')

ax.axis( (646917., 647721, 4185492., 4186106.) )

##

f=iz.as_raster()
# f.write_gdal("zhang-dem-bidir.tif")
# f.write_gdal("zhang-dem.tif")

# at least the bidirectional approach sufferrs from too-high
# a search_n.  Since it doesn't need a large number of samples
# to force a look both ways, a high number means that it's
# more likely to pick up lateral samples, particularly in sparse
# areas.

##

fig=plt.figure(12)
fig.clf()
ax=fig.add_subplot(1,1,1)

img=f.plot(ax=ax,cmap=cmap)

# f.plot_hillshade(z_factor=4,ax=ax)

iz.t_net.plot_nodes(mask=samples,color='k',ax=ax,alpha=0.3)

ax.axis( (647100, 647213, 4185804., 4185891.) )

##

# is it possible to search downstream and upstream separately?
# then we choose search_n/2 samples from downstream, search_n/2
# from upstream, and run IDW on the combined set?

# something is a little off - there is odd vertical striping now.

fig=plt.figure(13)
fig.clf()
ax=fig.add_subplot(1,1,1)

#f.plot(ax=ax,cmap=cmap)

#p=[647403, 4185689]
#p=[647560.8638027328, 4185545.21409829]
p=[647133, 4185860]
n=iz.t_net.select_nodes_nearest(p)

iz.t_net.plot_nodes(mask=[n],color='m',ax=ax)

# See which samples it uses:
self=iz
samples=np.nonzero( np.isfinite(self.t_net.nodes['value']) )[0]
#--
if 1:
    def fwd_weight(j,dir):
        Td=dir*self.t_net.edges['Td'][j]
        if Td>=0:
            return Td
        else:
            return np.nan

    def rev_weight(j,dir):
        Td=-dir*self.t_net.edges['Td'][j]
        if Td>=0:
            return Td
        else:
            return np.nan

    results_fwd=self.t_net.shortest_path(n,samples,edge_weight=fwd_weight,directed=True,
                                         return_type='cost',max_return=self.search_n//2)
    results_rev=self.t_net.shortest_path(n,samples,edge_weight=rev_weight,directed=True,
                                         return_type='cost',max_return=self.search_n//2)

# results are lists of (n,cost)
results=results_fwd+results_rev  # concatenate

r_nodes,r_costs = zip(*results) # "transpose" results
r_nodes=list(r_nodes) # tuple causes problems, make it a list.
result_values=self.t_net.nodes['value'][r_nodes]

iz.t_net.plot_nodes(mask=samples,color='k',ax=ax,alpha=0.3)
all_values=np.zeros(iz.t_net.Nnodes())
all_values[r_nodes]=r_costs
iz.t_net.plot_nodes(mask=r_nodes,values=all_values,ax=ax,cmap='jet',sizes=40)

# the forward samples are wrong, all in a column
zoom=(647338, 647453, 4185619, 4185735)
# zoom=(647398, 647408, 4185684, 4185691)

ax.axis(zoom)
##
# bidirectional is working.
# though it is now more speckled?

##

# What about keeping track of the location of the samples more?
#  so as we traverse the network to find samples, rather than just
# summing up per-edge distances, we sum vectors with respect to
# the local velocity / ellipse
# so each sample comes back not with just a distance, but instead
# with a measure that we went 10 units parallel, and 2 units
# to the right?

# pros: less diffusion by grid
# cons: could end up side stepping an area?
#       probably some deeper issues
#   is the distance field continuous? yes.  smooth?
#     maybe not.  there might be two distinct paths from n to a
#     particular node, leading to ridges.
