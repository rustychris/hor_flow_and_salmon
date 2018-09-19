# See what is different between the IC I'm specifying and what they come up with.
import xarray as xr
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
from stompy.plot import plot_utils

##

#ds_mine=xr.open_dataset('runs/steady_001/Estuary_IC.nc')
#ds_out =xr.open_dataset('runs/steady_001/Estuary_SUNTANS.nc_0000.nc')
#ds_out=xr.open_dataset('runs/steady_001/Estuary_SUNTANS.nc.nc.0')
ds_out=xr.open_dataset('runs/steady_001/backup.nc')


##

# g_mine=unstructured_grid.UnstructuredGrid.from_ugrid(ds_mine)
g_out= unstructured_grid.UnstructuredGrid.from_ugrid(ds_out)

##

dv=ds_out.dv.values
eta_out=ds_out.eta.isel(time=0).values
eta_mine=ds_mine.eta.isel(time=0).values

zoom=(648521.6040796808, 649494.3142301519, 4183322.778484539, 4184047.604435374)

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1,sharex=True,sharey=True)

# difference in eta
#ccoll=g_out.plot_cells(values=eta_out-eta_mine,clip=zoom,cmap='seismic')
#ccoll.set_clim([-0.01,0.01])

# output eta vs voro depth
ccoll=g_out.plot_cells(values=eta_out+dv,clip=zoom,cmap='seismic',ax=ax)
ccoll.set_clim([-0.01,0.01])

# # my dv vs. output dv:
# ccoll=g_out.plot_cells(values=ds_mine.dv.values-dv,
#                        clip=zoom,
#                        cmap='seismic')
# ccoll.set_clim([-2.0,2.0])

# # input dv vs. flat initial condition (-7.8m)
# ccoll=g_out.plot_cells(values=(-ds_mine.dv.values) - (-7.8),
#                        clip=zoom,
#                        cmap='seismic')
# ccoll.set_clim([-0.01,0.0])

# output bathy:
#ccoll_out =g_out.plot_cells(values=-ds_out.dv.values,clip=zoom,
#                       cmap='jet',ax=ax)
#ccoll_mine=g_out.plot_cells(values=-ds_mine.dv.values,clip=zoom,
#                       cmap='jet',ax=axs[1])
#axs.set_title('-ds_out.dv')
#axs[1].set_title('-ds_mine.dv.values')
plot_utils.cbar(ccoll)

ax.axis('equal')
ax.axis(zoom)

# near_clip=(648916, 648937, 4183691, 4183711)
near_clip=(648927, 648948, 4183691, 4183711)
g_out.plot_edges(clip=near_clip,labeler='id')
g_out.plot_cells(clip=near_clip,labeler=lambda i,r:str(i),facecolor='none')
plt.setp(ax.texts,ha='center')
ax.axis(near_clip)

##

# There is some wacky stuff going on with the coordinates.  wtf?


##
j=239604
nc1,nc2=g_out.edge_to_cells(j)

ds_out.U.isel(Ne=j,Nk=0) # up to 0.4 m/s

# UpdateDZ shows etop ending up as 1, but ds_out has it as 0, and
# velocities on the edge.  caching???

# the output has eta values both higher and lower than the initial condition.
# output eta is never lower than dv.
# is dv getting modified? yes. it is up to 2.0m deeper?
# (ds_mine.dv.values - ds_out.dv.values).min()

# Looks like areas which are not contiguous with the rest of the domain are getting
# bumped up in the bathy,
# and maybe that is leading to bogus velocity?

# my bathy still needs to be hard clipped to stay positive-down,
# and there is also some kind of min depth being applied which looks like
# at least 2m?

# my bathy is now being clipped at min depth, seems okay...
# model bathy output is clipped at -2 ??
# depths.data-voro definitely has values like 0.91, but
# ds_out.dv has min value of 2.003836

# model says something about AB=2 ?
#

# Seems that fixdzz has some bad logic, and was screwing up the
# dv?

##

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

ecoll=g_out.plot_edges(values=ds_out.dzf.isel(time=-1,Nk=0).values,
                       ax=ax)
eta_out=ds_out.eta.isel(time=-1).values
dv=ds_out.dv.values
wc_depth=eta_out+dv # min value of 1mm
ccoll=g_out.plot_cells(values=eta_out+dv,clip=zoom,cmap='seismic')
ccoll.set_clim([-0.01,0.01])


ax.axis('equal')
ecoll.set_lw(3.)
ecoll.set_cmap('jet')
plot_utils.cbar(ecoll)

# looks like dzf is set to 1mm, too, in many places.
# if these edges or cells are inactive, should they read 1mm,
# or 0??
# active is set in phys.c, but only used for scalar transport
# SetFluxHeight depends on etop to figure out wet/dry.
# 1. there is a wet/dry example.  compare suntans.dat to that.
#
# 2. Add some output to see whether these edges have etop==Nke[j],
#   etop is *never* set to Nke.  in my run, it is always 0, for
# all time steps all edges.

# trunk suntans sets etop to the lesser of the adjacent etops,
# which means that if there is a wet cell (ctop=0) and a dry cell
# (ctop=1), then it will get the wet cell.
# my old code instead sets etop to the upwind cell, or failing that
# the cell with a higher freesurface.
# is it possible to just use the cell with a higher free surface?
#

# Hmm - I changed the code to use ctop of the higher freesurface, but
# it is still showing etop=0, and nonzero velocity after the first
# step.  it's an improvement, as it avoids the higher velocities
# on the first step, but how is it getting wet?

# Is somebody else overwriting etop??
# seems like at the end of UpdateDZ, etop[j=239604]==1
# but just before it is written out to netcdf, it is 0.
# no obvious places in phys.c where it would get overwritten.

# revisiting this
# at the time of writing etop out in mynetcdf.c, some entries
# of grid->etop, like grid->etop[0], grid->etop[1], are 1, signalling
# a dry edge
# but grid->etop[239604] => 0
# and grid->etop[256046] => 0

# hmm - at the end of UpdateDZ, those are also 0.


##

# tight bend with some ill-behaved triangles
zoom=(645298.3114121656, 645468.0365386433, 4187163.385814458, 4187289.8584087044)
# g_sun=unstructured_grid.SuntansGrid('runs/steady_001')
# zoom=(648862.0417518547, 649132.9210624704, 4183453.5061032367, 4183655.354879855)
# tight zoom on western wetting front
zoom_in=(648937.5281613761, 648960.6659755674, 4183518.963843191, 4183538.9102347353)

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g_out.plot_edges(ax=ax,
                 clip=zoom,
                 lw=0.5,color='k')
ax.axis('equal')
g_out.plot_edges(ax=ax,
                 mask=[85492],
                 lw=3.5,color='r')

## 
# one issue is that nodes.dat.0 is written with only 6 sig figs, which is bad for
# utm coordinates.
# I think this becomes an issue when -g and -s are in separate invocations

# Plot nonzero velocities on edges
# first time step is relatively chill, max of 0.025 m/s
# second time step is bad.
U=ds_out.U.isel(time=1,Nk=0).values

ecoll=g_out.plot_edges(ax=ax,clip=zoom,mask=np.abs(U)>1e-5,lw=4,values=U,cmap='seismic')
ecoll.set_clim([-0.5,0.5])

g_out.plot_edges(ax=ax,clip=zoom_in,labeler='id')
g_out.plot_cells(ax=ax,clip=zoom_in,labeler=lambda i,r:str(i),facecolor='none')
ax.axis(zoom_in)
plt.setp(ax.texts,ha='center')

# j=246601
# cells: 120072, 120070
# at time=1, eta[c=120070]=-7.8   grid->grad[2*j]
#            eta[c=120072]=-5.99606  grid->grad[2*j+1]
# dv is 8.169885 and 5.99606 respectively.
# UpdateDZ is called with option=-1, then 1, then twice option=0, and the third time around it
# decides that this edge is wet.

# follow the logic around UpdateDZ:946
#   phys->h[120070] => -7.8000000000267065
#   phys->h[120072] => -5.9960599998717869
# but grid->ctop[120072]=0.  but it shouldn't.
# grid->dzz[120072][0] => 1.2821299577581158e-10
# because this is compared to 1.1*DRYCELLHEIGHT, which is 1.1e-10
# so who increased 120072's dzz?
#  first time through with option=0, it's legit, seems to already
# be propped up to DRYCELLHEIGHT
# dzz ends up  1.000000082740371e-10
# second time, 1.000000082740371e-10
# third time, phys->h comes in a bit higher.
# who could be doing this to h?
#   maybe the check around phys.c:3041 needs some hysteresis?
#   looks like we'll go through, prop up a cell, and mark it in active.
#   but then the next time, it wouldn't be so low, and we mark it active.
# presumably UPredictor is updating this freesurface
#  depends on etop being set to deactivate edges.
# what are the other edges of 120072 ?
#   they are all dry first time around, and 120072 is dry,
#   still at its propped up height.
# second time, those edges are still dry, and 120072 is still
#   at its CLAMPHEIGHT
# third time, its height has gone up, but none of the edges appear
#  to be wet. none of the dry neighbor cells are active.
# is is just a tolerance thing?  epsilon for CG was 1e-10, 
# and DRYCELLHEIGHT also 1e-10.  Offhand I don't know the units
# of epsilon, but that seems suspicious
# why does UpdateDZ get called twice per step?
# making epsilon half as large decreases the fake increase in
# in h by maybe 20%.

# with drycellheight at 2e-5, it was stable at 0.1
# runs at dt=1s for 19s, but then blowup...
# blow up maybe at a tight bend with bad triangles on Old River
# it's a pretty bad pair of triangles, and probably made worse by the
# rounding.
# the blowup is on , but the real culprit might be 76266
# the center are 0.30m away from each other

