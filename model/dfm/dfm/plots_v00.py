import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import xarray as xr
from stompy.model.delft import dfm_grid
from stompy.plot import plot_utils
import stompy.plot.cmap as scmap


##

ds=xr.open_dataset('runs/hor_002/DFM_OUTPUT_flowfm/flowfm_0000_20120801_000000_map.nc')

g=dfm_grid.DFMGrid(ds)

##

# Free surface - looks fine.
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

ccoll=g.plot_cells(ax=ax,values=ds.s1.isel(time=-1),cmap='jet')
ccoll.set_clim([1.9,2.5])

plot_utils.cbar(ccoll)
ax.axis('equal')

##

# Velocity magnitude
plt.figure(2).clf()
fig,ax=plt.subplots(num=2)

ucxa=ds.ucxa.isel(time=-1).values
ucya=ds.ucya.isel(time=-1).values

vmag=np.sqrt(ucxa**2 + ucya**2)

ccoll=g.plot_cells(ax=ax,values=vmag,cmap='jet')
plot_utils.cbar(ccoll)
ax.axis('equal')

# Max out around 0.4 m/s, with more like 0.2 m/s at the junction.

##

# Is it at steady state?

# upstream of junction
xy=(647300, 4185800)
c=g.select_cells_nearest( xy )

c_ucxa=ds.ucxa.isel(nFlowElem=c).values
c_ucya=ds.ucya.isel(nFlowElem=c).values

c_mag=np.sqrt(c_ucxa**2 + c_ucya**2)

plt.figure(3).clf()
fig,ax=plt.subplots(num=3)
ax.plot(ds.time.values, c_mag)

##

# The 3D run is roughly at steady state by 4h, with a little
# bit of adjusting / sloshing evident.

# Extract some transects.
# First, read Ed's transects:
untrim_sections_fn="../../untrim/ed-steady/section_hydro.txt"

from stompy import utils
utils.path("../../../field/adcp/")

import read_untrim_section

untrim_sections=read_untrim_section.section_hydro_to_dss(untrim_sections_fn)

##

# Find 7B for a nice comparison.
sec=[ sec for sec in untrim_sections if sec.attrs['name']=='7B'][0]

##

import stompy.model.delft.dflow_model as dfm
from stompy import xr_utils
import six
six.moves.reload_module(xr_utils)
six.moves.reload_module(dfm)

line_xy=np.c_[sec.x_utm.values, sec.y_utm.values]

prof=dfm.extract_transect(ds.isel(time=-1),line=line_xy,dx=3,grid=g)

xr_utils.bundle_components(prof,'U',['Ve','Vn'],'en',['E','N'])
xr_utils.bundle_components(prof,'U_avg',['Ve_avg','Vn_avg'],'en',['E','N'])

##

untrim_sec=xr_transect.section_hydro_to_transect(untrim_sections_fn,name="7B")

##

aprof=prof.copy()

aprof.attrs['name']=sec.attrs['name']
aprof.attrs['filename']='hor002'
aprof.rename( {'s1':'z_surf'}, inplace=True)
prof_adcp=read_untrim_section.ADCPSectionHydro(aprof)

##

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)

ax.quiver( prof.x_sample, prof.y_sample,
           prof.U_avg.sel(en='E'), prof.U_avg.sel(en='N'))

##

# Make sure the new plot code does something similar:
plt.figure(3).clf()
fig,axs=plt.subplots(2,1,num=3,sharex=True,sharey=True)

coll_u=xr_transect.plot_scalar(prof,prof.U.sel(en='E'),ax=axs[0])
coll_v=xr_transect.plot_scalar(prof,prof.U.sel(en='N'),ax=axs[1])

plt.setp([coll_u,coll_v], clim=[-0.5,0.5],cmap=cmap_sym)
for ax,coll,label in zip(axs,
                         [coll_u,coll_v],
                         ["U (m/s","V (m/s)"]):
    plt.colorbar(coll,ax=ax,label=label)
plt.setp(axs,facecolor='0.7')

##
from stompy import xr_transect
six.moves.reload_module(xr_transect)

prof['Q']=(),xr_transect.Qleft(prof)

xr_transect.add_rozovski(prof,'U','Uroz')

##

# Scatter with real z coordinate
plt.figure(4).clf()
fig,axs=plt.subplots(2,1,num=4,sharex=True,sharey=True)

x,y,Cu,Cv=xr.broadcast(prof.d_sample,prof.z_ctr,
                       prof.Uroz.isel(roz=0),prof.Uroz.isel(roz=1))
coll_u=plot_utils.pad_pcolormesh(x,y,Cu,ax=axs[0])
coll_v=plot_utils.pad_pcolormesh(x,y,Cv,ax=axs[1])


coll_u.set_clim([0,1.0])
coll_u.set_cmap(cmap_mag)
coll_v.set_clim([-0.1,0.1])
coll_v.set_cmap(cmap_sym)

for ax,coll,label in zip(axs,
                         [coll_u,coll_v],
                         ["U (m/s","V (m/s)"]):
    plt.colorbar(coll,ax=ax,label=label)

plt.setp(axs,facecolor='0.7')

##

# should push this to the read_untrim_section code
sec=sec.rename({
    'x_utm':'x_sample',
    'y_utm':'y_sample',
    'z':'z_ctr',
    'cell':'laydim'
})

##
# Is it really that hard to get ADCPy working?
# no, but it's not the right data model for any of the models
# or the Sontek ADCP.
six.moves.reload_module(xr_transect)

# Do this stuff again, but with the untrim data:
xr_utils.bundle_components(sec,'U',['Ve','Vn'],'en',['E','N'])

sec['U_avg']=xr_transect.depth_avg(sec,'U')
sec['d_sample']=('sample',), utils.dist_along( np.c_[sec.x_sample,sec.y_sample])

##
plt.figure(12).clf()
fig,ax=plt.subplots(num=12)

ax.quiver( sec.x_sample, sec.y_sample,
           sec.U_avg.sel(en='E'), sec.U_avg.sel(en='N'))

##

# Scatter with real z coordinate
plt.figure(13).clf()
fig,axs=plt.subplots(2,1,num=13,sharex=True,sharey=True)

x,y,Cu,Cv=xr.broadcast(sec.d_sample,sec.z_ctr,
                       sec.U.sel(en='E'),
                       sec.U.sel(en='N'))
coll_u=plot_utils.pad_pcolormesh(x.values,y.values,Cu.values,ax=axs[0])
coll_v=plot_utils.pad_pcolormesh(x.values,y.values,Cv.values,ax=axs[1])

plt.setp([coll_u,coll_v], clim=[-0.5,0.5],cmap=cmap_sym)
for ax,coll,label in zip(axs,
                         [coll_u,coll_v],
                         ["U (m/s","V (m/s)"]):
    plt.colorbar(coll,ax=ax,label=label)

plt.setp(axs,facecolor='0.7')

# HERE - basic plot works..

