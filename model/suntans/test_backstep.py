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

# backstep_00: initial... clobbered.
# 01: nu_H 0.0 => 0.25
model.set_run_dir(os.path.join('runs',"backstep_01"),
                  mode='pristine')

model.run_start=np.datetime64('2017-03-01 00:00:00')
model.run_stop=np.datetime64('2017-03-01 04:00:00')

model.config['nonlinear']=1
model.config['stairstep']=0
model.config['thetaM']=-1
model.config['z0B']=0.001
model.config['nu_H']=0.25
model.config['Nkmax']=1

dt=2.5
model.config['dt']=dt
model.config['ntout']=int(900./dt)
model.config['ntoutStore']=int(3600./dt)

W=100
L=1000
dx_lon=5
dy_lat=3
depth_center=-5
depth_edge=-5
def bathy(X):
    y=(X[:,1]-W/2.0)/(W/2.0) # 0 center, 1 edge
    return depth_center+(y**2)*(depth_edge-depth_center)

g=unstructured_grid.UnstructuredGrid(max_sides=4)

g.add_rectilinear( p0=[0,0], p1=[L,W],
                   nx=int(L/dx_lon),ny=int(W/dy_lat))

g.add_cell_field('depth',bathy(g.cells_center()))


model.set_grid(g)
model.config['maxFaces']=4

model.z_offset=0

from shapely import geometry
feats=np.zeros( 2, [ ('name','O'),
                     ('geom','O') ] )
feats[0]['name']='left'
# HALF STEP:
feats[0]['geom']=geometry.LineString( [ [0,W/2], [0,W] ] )
feats[1]['name']='right'
feats[1]['geom']=geometry.LineString( [ [L,0], [L,W] ] )

model.add_gazetteer(feats)

Q_left=drv.FlowBC(name='left',Q=200.0)
h_right=drv.StageBC(name='right',z=0.0)

model.add_bcs([Q_left,h_right])

model.write()
model.partition()
#model.run_simulation()

