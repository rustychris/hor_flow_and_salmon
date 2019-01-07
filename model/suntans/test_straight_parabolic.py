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

# straight_00: initial...
# straight_01: towards a bernoulli test case.
# straight_03: gaussian bump, no lateral variation
# straight_04: same, but nonlinear=0
# straight_05: nonlinear=1, one cell wide
# straight_06: 3D
model.set_run_dir(os.path.join('runs',"straight_06"),
                  mode='pristine')

model.run_start=np.datetime64('2017-03-01 00:00:00')
model.run_stop=np.datetime64('2017-03-01 04:00:00')

model.config['nonlinear']=1
model.config['stairstep']=1
model.config['thetaM']=-1
model.config['z0B']=0.000001
model.config['nu_H']=0.0
model.config['Nkmax']=25
model.config['wetdry']=1

dt=1.0
model.config['dt']=dt
model.config['ntout']=int(900./dt)
model.config['ntoutStore']=int(3600./dt)

W=5
L=1000
dx_lon=5
dy_lat=5
depth_center=-5
depth_edge=-5
def bathy(X):
    y=(X[:,1]-W/2.0)/(W/2.0) # 0 center, 1 edge
    z=depth_center+(y**2)*(depth_edge-depth_center)
    if 1: # add a bump in the middle of the domain
        bump=np.exp(-((X[:,0]-L/2)/50)**2)
        z+=1.5*bump
    return z

g=unstructured_grid.UnstructuredGrid(max_sides=4)

g.add_rectilinear( p0=[0,0], p1=[L,W],
                   nx=int(L/dx_lon),ny=1+int(W/dy_lat))

g.add_cell_field('depth',bathy(g.cells_center()))

model.set_grid(g)
model.config['maxFaces']=4

model.z_offset=0

from shapely import geometry
feats=np.zeros( 2, [ ('name','O'),
                     ('geom','O') ] )
feats[0]['name']='left'
feats[0]['geom']=geometry.LineString( [ [0,0], [0,W] ] )
feats[1]['name']='right'
feats[1]['geom']=geometry.LineString( [ [L,0], [L,W] ] )

model.add_gazetteer(feats)

Q_left=drv.FlowBC(name='left',Q=20.0)
h_right=drv.StageBC(name='right',z=0.0)

model.add_bcs([Q_left,h_right])

model.write()

if 0:
    # Add spatially variable roughness
    z0B=np.zeros( (1,model.grid.Nedges()), np.float64)
    ec=model.grid.edges_center()
    # 0...1000
    z0B[0,:]=0.001 + 0.1*(ec[:,1]>W/2)
    model.ic_ds['z0B']=('time','Ne'),z0B
    model.write_ic_ds()


model.partition()
model.sun_verbose_flag="-v"
model.run_simulation()

##

ds=xr.open_dataset(os.path.join(model.run_dir,'Estuary_SUNTANS.nc.nc.0'))


plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

x=ds.xv.values
ord=np.argsort(x)
# Eta
ax.plot( x[ord], ds.eta.isel(time=-1).values[ord],color='r',label='eta')

# bed
ax.plot( x[ord], -ds.dv.values[ord], color='k', label='bed')

# bernoulli
eta=ds.eta.isel(time=-1).values
u=ds.uc.isel(time=-1,Nk=0).values
bern=eta + u**2/(2*9.8)
ax.plot( x[ord], bern,color='b',label='bern')

ax.legend()

ax.axis( (368., 635., -0.12, 0.12) )
ax.grid(1)



