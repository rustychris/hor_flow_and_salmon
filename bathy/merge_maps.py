"""
A homegrown map merge, as the DFM one seems to have issues.
"""

import xarray as xr
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid

import glob
import re
import numpy as np

##

output_dir="../model/dfm/dfm/runs/hor_002/DFM_OUTPUT_flowfm"

def partitioned_map_files(output_dir,period=None):
    map_files=glob.glob( os.path.join(output_dir,"*_map.nc") )

    partitioned_map_files=[]
    procs=[]
    for map_file in map_files:
        m=re.match( r'(.*)_([0-9]{4})_([0-9]{8}_[0-9]{6})_map.*', map_file )
        if not m:
            continue
        run_name=m.group(1)
        proc=m.group(2)
        this_period=m.group(3)
        if period is None:
            period=this_period
        else:
            if period!=this_period:
                continue # from a different period of output
        partitioned_map_files.append(map_file)
        procs.append( int(proc) )
    order=np.argsort( procs )
    # make it nice and ordered by subdomain
    return [ partitioned_map_files[i] for i in order]

map_files=partitioned_map_files(output_dir)

dss=[xr.open_dataset(fn) for fn in map_files]
nproc=len(dss)

##

# Not really ready for transcribing gigabytes.  Current need is for a single
# timestep, go with that.
dss=[ds.isel(time=-1) for ds in dss]

##

# Construct the global grid
grids=[dfm_grid.DFMGrid(ds) for ds in dss]

node_maps={}
edge_maps={}
cell_maps={}

gg=unstructured_grid.UnstructuredGrid(max_sides=8)

for proc in range(nproc):
    print("Merging grid %d"%proc)
    n_map,e_map,c_map = gg.add_grid(grids[proc],merge_nodes='auto')
    node_maps[proc]=n_map
    edge_maps[proc]=e_map
    cell_maps[proc]=c_map

# I'm ending up with 2 cells more than this file:
# g_orig=dfm_grid.DFMGrid('../model/dfm/dfm/runs/hor_002/grid_net.nc')
# but the same number of cells, edgse and nodes as this: bueno!
# g_int=dfm_grid.DFMGrid('../model/dfm/dfm/runs/hor_002/DFM_interpreted_idomain_grid_net.nc')

##

# for starters, build it all at once.  may have to redesign to do this more incrementally
# in the future

gds=gg.write_to_xarray(mesh_name=mesh_topology.name,
                       node_coordinates=mesh_topology.node_coordinates,
                       face_node_connectivity=mesh_topology.face_node_connectivity,
                       edge_node_connectivity=mesh_topology.edge_node_connectivity,
                       node_dimension=mesh_topology.node_dimension,
                       edge_dimension=mesh_topology.edge_dimension,
                       face_dimension=mesh_topology.face_dimension)

gds.attrs.update( dss[0].attrs )

##

data_vars_to_copy=[]
for vname in dss[0].data_vars:
    v=dss[0][vname]
    # scan dimensions, skip things we're not ready for:
    handle=True
    if 'enclosure' in vname:
        continue
    for dim in v.dims:
        if ( ('Enclosure' in dim) or
             ('Contour' in dim) ):
            handle=False
            break
    if not handle:
        continue
    if v.attrs.get('cf_role','') == 'mesh_topology':
        mesh_topology=v
        # mesh topology is just names, and can be copied directly
        gds[vname]=v
        continue
    if vname=='projected_coordinate_system':
        # likewise - just copy
        gds[vname]=v
        continue
    if ( vname==mesh_topology.edge_node_connectivity or
         vname==mesh_topology.face_node_connectivity or
         vname in mesh_topology.node_coordinates):
        continue # part of mesh handling
    if vname in ['NetElemLink','BndLink','ElemLink','FlowLink']:
        # These shouldn't be copied directly, but may be handled manually
        continue
    if vname in ['FlowElemDomain','FlowLinkDomain','FlowElemGlobalNr']:
        # These are related to subdomains, and don't make sense in the
        # global file.
        continue

    data_vars_to_copy.append(vname)

##

# Are net elements and flow elements the same?  Checks out so far.
# This simplifies the translation
for g,ds in zip(grids,dss):
    cc=g.cells_center()
    assert np.allclose( ds.FlowElem_xcc.values, cc[:,0])
    assert np.allclose( ds.FlowElem_ycc.values, cc[:,1])

##

# Flow Links: assume that net elements (faces/cells) are the same as flow elements
# these are not represented directly in the grid, so come back now to create
# the map for FlowLinks
flowlink_maps={}
vname='FlowLink' # this maps each flow link to a pair of cells
flowlink_dim=dss[0][vname].dims[0] # used in copying.

# Could assume that flow links are a subset of netlinks
# net_link_is_flow_link=np.zeros( gg.Nedges(),np.bool8)

xy_to_flow_link={} 
flow_links=[]
boundary_elements=[]

for proc in range(nproc):
    print("Marking flow links from proc %d"%proc)
    fl_local=dss[proc][vname].values

    link_x=dss[proc]['FlowLink_xu'].values
    link_y=dss[proc]['FlowLink_yu'].values

    flowlink_maps[proc]=my_map=np.zeros( len(fl_local), np.int32 )-1

    nelems=len(dss[proc].nFlowElem)
    assert nelems==grids[proc].Ncells() # sanity check

    for j,loc_elems in enumerate(fl_local):
        k=(link_x[j],link_y[j])
        if k not in xy_to_flow_link:
            e1,e2=loc_elems # 1-based!
            def loc_to_global(elem):
                # elem is a 1-based flow element index local to proc
                # could be a boundary element, denoted by being greater
                # than the number of local elements
                # return a 1-based global element number, but switching to
                # boundary elements being negative, -1, decreasing
                assert elem>0,"Thought that was always 1-based"
                if elem<=nelems: # a proper element
                    return 1+cell_maps[proc][elem-1]
                else:
                    boundary_elements.append( (proc,j,elem) )
                    return -len(boundary_elements)
            xy_to_flow_link[k]=len(flow_links)
            flow_links.append((loc_to_global(e1), loc_to_global(e2)))
        my_map[j]=xy_to_flow_link[k]

flow_links=np.array(flow_links) # max is 126740
assert flow_links.max()==gg.Ncells() # is that optimistic?  works so far.
# Now revert to DFM boundary numbering
boundary=flow_links<0
flow_links[boundary] = -flow_links[boundary] + gg.Ncells()

gds[vname]=dss[0][vname].dims,flow_links

##
elem_flags=np.zeros(gg.Ncells(),np.int32)

# Remove ghosts from maps, at least for cells
for proc in range(nproc):
    flow_elem_dom=dss[proc].FlowElemDomain.values
    # a little defensive check to make sure this is 0-based.
    assert flow_elem_dom.min()>=0
    assert flow_elem_dom.max()<=nproc-1
    non_local=flow_elem_dom!=proc
    cell_maps[proc][non_local]=-1

    # And make sure that all flow elements still have somebody
    # mapped to them.
    valid=cell_maps[proc]>=0
    elem_flags[ cell_maps[proc][valid] ]=1
assert np.all(elem_flags==1)

##

mappers={}
mappers[mesh_topology.node_dimension]=node_maps
mappers[mesh_topology.edge_dimension]=edge_maps
mappers[mesh_topology.face_dimension]=cell_maps
mappers[flowlink_dim]=flowlink_maps
# Flow elements and net elements are the same:
mappers['nFlowElem']=mappers[mesh_topology.face_dimension]

# can get a good guess of the sizes by just looking
# for the largest value in the maps
shapes={}
for k in mappers:
    shapes[k]=1+max( [mappers[k][m].max() for m in mappers[k]])

assert shapes[mesh_topology.node_dimension]==gg.Nnodes()
assert shapes[mesh_topology.edge_dimension]==gg.Nedges()
assert shapes[mesh_topology.face_dimension]==gg.Ncells()

##

for vname in data_vars_to_copy:
    v=dss[0][vname]
    print("Copying variable %30s"%vname, end="")
    dims=v.dims

    shape=list(v.shape)

    # figure out the shape of the new thing:
    for di,d in enumerate(dims):
        if d in shapes:
            shape[di]=shapes[d]
        elif d in gds.dims:
            shape[di]=len(gds[d])
        else:
            print(" [dim %s will pass through]"%d,end="")

    new_val=np.zeros(shape,v.dtype)
    if len(shape)==0:
        # could check to make sure this is the same across procs
        gds[vname]=v
        print()
        continue

    new_val[:] = np.nan

    for proc in range(nproc):
        print(" %d"%(proc),end="")
        old_sel=[slice(None)]*len(dims)
        new_sel=[slice(None)]*len(dims)

        for di,d in enumerate(dims):
            mapper=None
            if d in mappers:
                mapper=mappers[d][proc]
            if mapper is not None:
                # this takes care of per-domain items which didn't make the cut
                old_sel[di]=(mapper>=0)
                # and this does the real mapping to global ids
                new_sel[di]=mapper[ mapper>=0 ]

        # This values call is going to get painful with time
        new_val[tuple(new_sel)] = dss[proc][vname].values[old_sel]
    print(" done")
    gds[vname]=v.dims,new_val

##

fn='merged_map.nc'
os.path.exists(fn) and os.unlink(fn)
gds.to_netcdf('merged_map.nc')
gds.close()

##


