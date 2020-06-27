import six
import logging
log=logging.getLogger('hor_sun')
log.setLevel(logging.INFO)

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from stompy import filters, utils
from stompy.spatial import wkb2shp
from stompy.grid import unstructured_grid

from stompy.model.suntans import sun_driver as drv
import stompy.model.delft.dflow_model as dfm

six.moves.reload_module(utils)
six.moves.reload_module(dfm)
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(drv)

model=drv.SuntansModel()
model.projection="EPSG:26910"

model.num_procs=1
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.mpi_bin_dir="/usr/bin"

model.load_template("sun-template.dat")

# bend_00: initial...
# bend_01: 3D
# bend_02: 3D, no bathy variation
# bend_03: 3D, bathy, and Kh=5.
#   50m wide channel, flow O(1 m/s), would like it to mix within 100m,
#   T=100s, K=25^2/100 => 6.25 m2/s
# bend_04: bump up nu 1e-6 => 1e-5
# bend_05: 0.01...
# bend_06: 0.02 and 2m2/s
# bend_07: 0.02 and 0m2/s
# bend_08: 0.02 and 0.2 m2/s
# bend_09: 0.02 and 0.0 m2/s, z0B to 0.01
# bend_10: using parabolic eddy viscosity.  dial down nu
# bend_11: like 10, but rotate the channel by 90deg
# bend_12:              rotate the channel by 45deg
# bend_13:              rotate the channel by 180deg
model.set_run_dir(os.path.join('runs',"bend_13"),
                  mode='pristine')

model.run_start=np.datetime64('2017-03-01 00:00:00')
# seems that this spins up by 2.5h
model.run_stop=np.datetime64('2017-03-01 02:30:00')

model.config['nonlinear']=1
model.config['stairstep']=0
model.config['thetaM']=-1
model.config['z0B']=0.01
model.config['nu_H']=0.0
model.config['nu']=1e-5
model.config['Nkmax']=20
model.config['turbmodel']=10

dt=0.5
model.config['dt']=dt
model.config['ntout']=int(900./dt)
model.config['ntoutStore']=int(3600./dt)

W=50
L=300
rotate=180.0
dx_lon=5
dy_lat=3
depth_center=-5
depth_edge=-1
if model.run_dir.endswith("_02"):
    depth_edge=depth_center

def bathy(X):
    y=(X[:,1]-W/2.0)/(W/2.0) # 0 center, 1 edge
    return depth_center+(y**2)*(depth_edge-depth_center)

six.moves.reload_module(unstructured_grid)
g=unstructured_grid.UnstructuredGrid(max_sides=4)


# A centerline:
sec0=np.array([[0,W/2],[L,W/2]])
s=np.linspace(0,np.pi/2,40)[1:] # skip first point
x_arc=np.cos(s)
y_arc=np.sin(s)

bend_r=W # radius of bend
bend_ctr=sec0[-1] + np.array([0,bend_r])
sec1=bend_ctr + bend_r*np.c_[ y_arc, -x_arc]

sec2=np.array( [sec1[-1],
                sec1[-1] +np.array([0,L] ) ] )

centerline=np.concatenate([sec0,sec1,sec2])
from stompy.spatial import linestring_utils
centerline=linestring_utils.resample_linearring(centerline,dx_lon,closed_ring=0)


def profile(x,s):
    return np.linspace(-W/2.,W/2.,int(W/dy_lat))

ret=g.add_rectilinear_on_line(centerline,profile)

y=g.cells['d_lat']/(W/2.0)
bathy=depth_center+(y**2)*(depth_edge-depth_center)
g.add_cell_field('depth',bathy,on_exists='overwrite')

if rotate!=0.0:
    g.nodes['x']=utils.rot(rotate*np.pi/180., g.nodes['x'] )



inflow= g.nodes['x'][ret['nodes'][0,:]]
outflow=g.nodes['x'][ret['nodes'][-1,:]]

if 1:
    plt.figure(1).clf()
    plt.plot(centerline[:,0],
             centerline[:,1],'g-')
    g.plot_edges(color='k',lw=0.5)
    ccoll=g.plot_cells(values=g.cells['depth'],cmap='jet')
    plt.colorbar(ccoll)
    plt.axis('equal')

# plt.figure(1).clf()
# g.plot_cells(values=g.cells['depth'],cmap='jet')
# plt.axis('equal')

model.set_grid(g)
model.config['maxFaces']=4

model.z_offset=0

from shapely import geometry
feats=np.zeros( 2, [ ('name','O'),
                     ('geom','O') ] )
feats[0]['name']='left'
feats[0]['geom']=geometry.LineString( inflow )
feats[1]['name']='right'
feats[1]['geom']=geometry.LineString( outflow )
model.add_gazetteer(feats)

Q_left=drv.FlowBC(name='left',Q=200.0)
h_right=drv.StageBC(name='right',z=0.0)

model.add_bcs([Q_left,h_right])

model.write()
model.partition()
#model.run_simulation()

