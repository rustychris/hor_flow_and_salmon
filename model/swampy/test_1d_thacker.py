"""
Test 1D version of the Thacker test case as a validation of 
subgrid.
"""

import stompy.grid.unstructured_grid as ugrid
import matplotlib.pyplot as plt
import SWAMPpy
import numpy as np
from stompy import utils
import six

##

six.moves.reload_module(SWAMPpy)

class SwampyThacker1D(SWAMPpy.SwampyCore):
    def set_initial_conditions(self,h=0.0):
        # allocates the arrays
        super(SwampyThacker1D,self).set_initial_conditions()

        # bed elevation, positive-down
        self.ic_zi[:]=-self.grd.cells['cell_depth']
        # water surface, positive up
        self.ic_ei[:]=np.maximum(h,-self.ic_zi)


W=20
L=1000
nx=100
h0=5
r0=L/3.
eta=0.1

omega=np.sqrt(2*9.8*h0)/r0

U=eta*r0*omega

dx=L/float(nx)
dt=0.5*dx/U

sim=SwampyThacker1D(dt=dt,ManN=0.0,theta=0.55)
#sim.get_fu=sim.get_fu_no_adv
sim.get_fu=sim.get_fu_Perot

sim.cg_tol=1e-11

g=ugrid.UnstructuredGrid(max_sides=4)
g.add_rectilinear([-L/2,0],[L/2,W],nx+1,2)
g.orient_edges()

def depth_fn(xy):
    return -h0 * (1-(xy[:,0]/r0)**2)
def eta_fn(xy,t):
    fs= eta * h0/r0*(2*xy[:,0]*np.cos(omega*t) - eta*r0*np.cos(omega*t)**2)
    bed=depth_fn(xy)
    return np.maximum(fs,bed)

g.add_node_field('node_depth',depth_fn(g.nodes['x']))
g.add_cell_field('cell_depth',depth_fn(g.cells_center()))

sim.set_grid(g)
sim.set_initial_conditions(h=eta_fn(g.cells_center(),t=0))

period=2*np.pi/omega

if 1:
    sim.last_plot=-1000000
    sim.plot_interval=period/20
    @utils.add_to(sim)
    def step_output(self,n,ei,**kwargs):
        if self.t-self.last_plot<self.plot_interval:
            return
        self.last_plot=self.t

        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)

        ax.plot(self.grd.cells_center()[:,0], -self.zi, 'b-o',ms=3)
        ax.plot(self.grd.cells_center()[:,0], self.ei, 'g-' )
        ax.plot(self.grd.cells_center()[:,0],
                eta_fn(g.cells_center(),self.t),color='orange')

        plt.draw()
        plt.pause(0.01)
    
(hi, uj, tvol, ei) = sim.run(tend=4*period)

#

fig=plt.figure(1)

fig.clf()

ax=fig.add_subplot(1,1,1)

ax.plot(g.cells_center()[:,0], -sim.zi, 'b-o',ms=3)
ax.plot(g.cells_center()[:,0], sim.ei, 'g-' )
ax.plot(g.cells_center()[:,0],
        eta_fn(g.cells_center(),sim.t),color='orange')

##

# generally working -
# get very occasional negative hjstar
# probably worth double-checking how that happens, even if it's just
# once per period.

# with longer time steps it is significantly damped.
# even with short timesteps, it's slightly damped and slightly lagged.
# ==> go back to theta-method

