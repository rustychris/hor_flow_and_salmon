import os.path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
from matplotlib.cm import jet
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import num2date, date2num
import stompy.grid.unstructured_grid as unstructured_grid
from rmapy.hydro import UgridNetCDF_hydro
import cPickle as pickle
import matplotlib.gridspec as gridspec
import xarray as xr
import shelve
from netCDF4 import Dataset, num2date
import pdb

def loadTransectShapeFile(shp): # from Steve's TransectFile.py
    from osgeo import ogr
    from shapely.wkb import loads

    driver = ogr.GetDriverByName('Esri Shapefile')
    dataset = driver.Open(shp, 0)
    layers = dataset.GetLayer()
    stationName = []
    pts = []
    for layer in layers:
        stationName.append(layer.GetField("name"))
        geom = loads(layer.GetGeometryRef().ExportToWkb())
        points = list(geom.coords)
        pts.append(points)
    numTransects = len(stationName)

    return stationName,pts

shp = r'R:\UCD\Projects\CDFW_Klimley\GIS\flow_lines.shp'
names, pts = loadTransectShapeFile(shp)

ncfile = r'R:\Hydro\SanJoaquin\cfg008\ptm_average.nc_0000.nc'
grd = unstructured_grid.UnstructuredGrid.from_ugrid(ncfile,'Mesh2')

bc_file = r'R:\Hydro\SanJoaquin\cfg008\cfg008_20180310\Estuary_BC.nc'
bc = xr.open_dataset(bc_file)
bc_flow = {}
# use names in shapefile
bc_flow['San_Joaquin_upstream_boundary'] = bc['boundary_Q'][:,0]
bc_flow['San_Joaquin_downstream_boundary'] = -bc['boundary_Q'][:,1]
#SJ_in = bc['boundary_Q'][:,0]
dnum_bc = bc['time']

ds = xr.open_dataset(ncfile)
#nc = Dataset(ncfile) # not currently used
dtime = ds['Mesh2_data_time'].values

flow_ts = {}
bc_names = ['San_Joaquin_upstream_boundary','San_Joaquin_downstream_boundary']
for ntran, name in enumerate(names):
# get edges 
    n1 = grd.select_nodes_nearest(pts[ntran][0]) # start node of transect
    n2 = grd.select_nodes_nearest(pts[ntran][1]) # end node of transect
    nodes_tran = grd.shortest_path(n1, n2, return_type='nodes')
    edges_tran = grd.shortest_path(n1, n2, return_type='edges')
    ecenters = grd.edges_center()[edges_tran]
    normals = grd.edges_normals(edges_tran)
    nedges = len(edges_tran)
    signs = np.zeros(nedges, np.float64)
    signed_normals = np.zeros_like(normals)
    for ne, edge in enumerate(edges_tran):
        enodes = grd.edges['nodes'][edge] 
        loc0 = np.where(nodes_tran == enodes[0])[0][0]
        loc1 = np.where(nodes_tran == enodes[1])[0][0]
        if loc1 == loc0 + 1:
            signs[ne] = -1.0
        else:
            signs[ne] = 1.0
        signed_normals[ne,:] = np.asarray([normals[ne,0]*signs[ne],
                                           normals[ne,1]*signs[ne]])

    h_flow_avg_edges = ds['h_flow_avg'].values[edges_tran,...]
    kbj = ds['Mesh2_edge_bottom_layer'].values[edges_tran,...]
    ktj = ds['Mesh2_edge_top_layer'].values[edges_tran,...]
    # vertical sum
    h_flow_avg_wcs = np.nansum(h_flow_avg_edges,axis=1)
    nsteps = len(dtime)
    tflow = np.zeros(nsteps, np.float64)
    dt_seconds = 30.*60.
    for nt in range(nsteps):
        # vertical sum
        print("nt = %d"%nt)
        for ne, edge in enumerate(edges_tran):
            tflow[nt] += h_flow_avg_wcs[ne,nt]*signs[ne]

    flow_ts[name] = tflow
    plt.ion()
    plt.show()
    t_offset = 15.*60
    plt.plot_date(dtime[:nsteps],tflow,'g-',label='calc all k')
    if name in bc_names:
        plt.plot_date(dnum_bc,bc_flow[name],'r-',label='boundary Q')
    dstart = datetime(2018,3,10)
    dend = datetime(2018,3,20)
    plt.gca().set_xlim([dstart,dend])
    plt.gca().set_ylabel('flow (cms)')
    plt.legend(loc='upper left')
    dayslocator = matplotlib.dates.DayLocator(bymonthday = [1,5,10,15,20,25,30])
    plt.gca().xaxis.set_major_locator(dayslocator)
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.gca().set_title(name)
    plt.savefig('%s.png'%name)
    plt.close()

shelve_dict = {}
pdb.set_trace()
shelve_dict['dtime'] = dtime
shelve_dict['names'] = names
shelve_dict['flows'] = flow_ts
#shelf = shelve.open('hydro_flows.shelve', protocol=2)
shelf = shelve.open('hydro_flows.shelve')
shelf.update(shelve_dict)
hpickle = open('hydro_pickle.pickle','wb')
pickle.dump(shelve_dict,hpickle)
hpickle.close()
