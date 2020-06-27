from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import SWAMPpy
import numpy as np
from stompy import utils
import six

##

six.moves.reload_module(SWAMPpy)

class SwampyHOR(SWAMPpy.SwampyCore):
    def set_grid(self,ug):
        """
        Set grid from unstructured_grid.UnstructuredGrid instance
        """
        self.grd=ug
        self.set_topology()

    def set_initial_conditions(self,h=0.0):
        # freesurface per cell -- allocate
        self.ic_ei=np.zeros(self.ncells, np.float64)
        # bed elevation, positive-down -- allocate
        self.ic_zi=-self.grd.cells['cell_depth']

        # set values
        self.ic_ei[:]=np.maximum(h,-self.ic_zi)

        # set_initial_conditions uses ds_i, us_i, us_j index arrays
        # but are those used outside set initial condition?
        # yes, yes they are.

    fig=None
    ax=None
    def prep_figure(self):
        plt.figure(1).clf()
        self.fig,self.ax=plt.subplots(num=1)
        self.ax.set_position([0,0,1,1])
        # self.grd.plot_edges(ax=self.ax,color='k',lw=0.5)
        # self.ax.axis('equal')
        self.ax.xaxis.set_visible(0)
        self.ax.yaxis.set_visible(0)

    def Xstep_output(self,n,ei,**kwargs):
        if self.fig is None:
            return
        if n%20:
            return

        self.prep_figure()
        ccoll=sim.grd.plot_cells(values=sim.hi,cmap='gray',ax=self.ax)
        cc=sim.grd.cells_center()
        ui = sim.get_center_vel(sim.uj)
        mags=utils.mag(ui)
        quiv=self.ax.quiver(cc[:,0],cc[:,1],ui[:,0],ui[:,1],mags,cmap=cm.rainbow,clim=[0,1.0])

        # plt.colorbar(ccoll)
        self.ax.quiverkey(quiv,0.1,0.1,0.5,"0.5 m/s")
        self.ax.axis('equal')
        self.fig.canvas.draw()
        plt.pause(0.001)


sim=SwampyHOR(dt=1.0,ManN=0.025)
sim.cg_tol=1e-10 # having a hard time converging.
g=unstructured_grid.UnstructuredGrid.from_ugrid("../suntans/snubby-with_bathy.nc")
# For testing, drop depth everywhere.
# this code isn't working for wetting and drying, so just nix all dry
# cells.
eta0=2.2
to_delete=np.nonzero(g.cells['cell_depth']>(eta0-0.5))[0]
for c in to_delete:
    g.delete_cell(c)
print("Deleted %d cells based on dryness"%len(to_delete))
to_delete=np.nonzero(g.edge_to_cells().max(axis=1)<0)[0]
for j in to_delete:
    g.delete_edge(j)
print("And %d edges"%len(to_delete))

g.renumber()
g.orient_edges()

sim.set_grid(g)
sim.set_initial_conditions(h=2.2)

sj_inflow=np.array( [[648045.3, 4185562.3],
                     [648035.2, 4185696.1]] )
sj_outflow=np.array( [[647513.1, 4186185.6],
                      [647542.6, 4186070.4]] )
or_outflow=np.array( [[646845.5, 4185762.2],
                      [646857.0, 4185842.4]] )

sim.add_bc( SWAMPpy.FlowBC(geom=sj_inflow,Q=220,ramp_time=1800) ) # 220
sim.add_bc( SWAMPpy.FlowBC(geom=sj_outflow,Q=-100,ramp_time=1800) )
sim.add_bc( SWAMPpy.StageBC(geom=or_outflow,h=2.2) )


sim.prep_figure()
plt.draw()

# speeding this up:
#   7.74s
# get_fu_Perot is 3.0s
#   calc_hjstar, hjbar, hjtilde, each: 0.4
# get_center_vel is 2s

##

%prun sim.run(tend=10)

## 
(hi, uj, tvol, ei) = sim.run(tend=7200.)  # final water surface elevation, velocity

##

ds=g.write_to_xarray()
ds['hi']=('face',), hi
ds['uj']=('edge',),uj
ds['ei']=('face',),ei

ui = sim.get_center_vel(sim.uj)

ds['ui']=('face','two',),ui

# ds.to_netcdf('frictionless.nc')
ds.to_netcdf('manning-0.025.nc')

##

plt.figure(2).clf()
mag=utils.mag(ui)

g.plot_cells(values=mag,cmap='jet',clim=[0,1])

##
import xarray as xr
obs=xr.open_dataset("../../field/adcp/040518_BT/040518_5BTref-avg.nc")

xy=np.c_[ obs.orig_x_sample.values, obs.orig_y_sample.values ]
xy=xy[np.isfinite(xy[:,0]),:]

#tran=model.extract_transect(xy=xy,time=-1,dx=2)

cells=[g.select_cells_nearest(pnt,inside=True)
       for pnt in xy]
cells=utils.remove_repeated(np.array([c for c in cells if c is not None]))

## 
tr=xr.Dataset()
cc=g.cells_center()
tr['x_sample']=('sample',),cc[cells,0]
tr['y_sample']=('sample',),cc[cells,1]
tr['eta']=('sample',),ei[cells]
tr['z_dz']=('sample','layer'), (ei[cells]+sim.zi[cells])[:,None]
tr['z_ctr']=tr.eta-0.5*tr.z_dz
tr['U']=('sample','layer','xy'), ui[cells,:][:,None,:]

##
from stompy import xr_transect
xr_transect.add_rozovski(tr)

plt.figure(2).clf()
fig,axs=plt.subplots(2,1,sharex=True,num=2)

xr_transect.plot_scalar(tr,tr.Uroz.isel(roz=0),ax=axs[0])
axs[1].plot( tr.d_sample, tr.Uroz.isel(roz=0,layer=0) )
axs[0].set_title('SWAMPY transect 5.  Downstream velocity. n=0.025')
axs[1].axis(ymin=0)
axs[1].set_ylabel('m/s')
fig.savefig('swampy-n0.025-transect_05.png')

##
fig=plt.figure(1)
fig.savefig('swampy-n0.025-planview.png')



