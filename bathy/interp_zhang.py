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
six.moves.reload_module(unstructured_grid)

##

import bathy
dem=bathy.dem()

##

class InterpZhang(object):
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
        self.clip=clip

        self.g=g
        self.uv=np.c_[u,v]

        self.build_net()
        self.set_velocity_on_net()
        self.calc_Td()

    def calc_Td(self):
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
            u_tan=abs(u*dx+v*dy)
            if u_tan!=0.0:
                Tab=(2*(dx**2 +dy**2))/u_tan
            else:
                Tab=np.nan
            self.t_net.edges['Td'][j]=Tab

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


class InterpZhangDual(InterpZhang):
    def build_net(self):
        """
        Build a pseudo grid that has edges for neighbors (like a dual, but including
        a larger neighborhood.
        """
        self.t_dual=self.g

        # transit time network
        self.t_net=unstructured_grid.UnstructuredGrid(extra_node_fields=[('value',np.float64),
                                                                         ('dual_cell',np.int32)
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
        cells=self.t_net.nodes['cell']
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


class InterZhangCartesian(InterpZhang):
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
        self.cart_map = self.t_dual.add_rectilinear( p0,p1,nx,ny )
        # cart_map gets ['cells'] => [nx-1,ny-1] array to cell index
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

        self.t_dual.renumber() # need to toss unused nodes, edges, too.

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

iz=InterZhangCartesian(g,
                       u=ds.ucxa.values,
                       v=ds.ucya.values,
                       clip=(646966, 647602, 4185504, 4186080),
                       dx=2,dy=2)

iz.set_samples(xyz_input)
iz.interpolate()

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
iz.t_dual.plot_cells(ax=ax)
iz.t_net.plot_edges(lw=0.3,color='k',ax=ax)
iz.t_net.plot_nodes(values=iz.t_net.nodes['int_value'],sizes=40,ax=ax)

ax.axis('equal')

##

plt.figure(11).clf()
fig,ax=plt.subplots(num=11)
ecoll=iz.t_net.plot_edges(values=iz.t_net.edges['Td'],lw=2.,ax=ax)
plot_utils.cbar(ecoll)
ax.axis('equal')

# iz.t_net.plot_nodes(values=t_net.nodes['value'],zorder=2)

##

fig=plt.figure(10)
fig.clf()
fig,ax=plt.subplots(num=10)

t_net=iz.t_net
t_net.plot_edges(ax=ax,color='k',lw=0.5,alpha=0.3)
t_net.plot_nodes(ax=ax,values=t_net.nodes['int_value'],sizes=40,cmap=cmap)

##

cell_values=np.zeros( iz.t_dual.Ncells(),'f8' )
cell_values[:]=np.nan
# value looks fine, but int_value is off.
cell_values[iz.t_net.nodes['dual_cell']]=iz.t_net.nodes['int_value']

plt.figure(11).clf()
fig.clf()
fig,ax=plt.subplots(num=11)

# iz.g.plot_edges(color='k',lw=0.2,clip=clip)
ccoll=iz.t_dual.plot_cells(values=cell_values,
                           mask=np.isfinite(cell_values),clip=clip,cmap=cmap,ax=ax)
ccoll.set_clim([-8,3])
plot_utils.cbar(ccoll)

ax.plot(xyz_input[:,0],xyz_input[:,1],'k.',ms=0.3)

ax.axis('equal')

ax.axis( (646917., 647721, 4185492., 4186106.) )

# The cartesian results look significantly worse.  Is velocity getting on the
# grid properly?
# fig.savefig('zhang-on-grid.png')

##
zoom=(646855, 647695, 4185473, 4186134)

plt.figure(12).clf()
fig,ax=plt.subplots(num=12)
iz.g.plot_cells(ax=ax,lw=0.1,facecolor='none',edgecolor='k',clip=zoom)

valid=(iz.t_net.nodes['u']!=0.0)|(iz.t_net.nodes['v']!=0.0)
ax.quiver( iz.t_net.nodes['x'][valid,0],
           iz.t_net.nodes['x'][valid,1],
           iz.t_net.nodes['u'][valid],
           iz.t_net.nodes['v'][valid],
           scale=10)
ax.axis('equal')

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
