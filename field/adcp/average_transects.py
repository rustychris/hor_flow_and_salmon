"""
Try using ADCPy to merge multiple transects.
"""
import os
import glob

from stompy import utils

utils.path( os.path.join( os.environ['HOME'],"src/ADCPy") )

## 

from adcpy import adcpy
import read_sontek

class ADCPSontekM9(adcpy.ADCPData):
    def __init__(self,rivr_file,**kwargs):
        super(ADCPSontekM9,self).__init__(**kwargs)
        
        self.rivr_file=rivr_file
        self.convert_sontek(self.rivr_file)
    def convert_sontek(self,rivr_file):
        self.ds=read_sontek.surveyor_to_xr(rivr_file)

        # Set ADCPData members from self.ds

        # Starts with z as a positive-down quantity
        # use round(...,4) to drop some FP roundoff trash
        z_in=self.ds.location.values
        min_z=round(np.nanmin(z_in),4)
        min_dz=np.round(np.nanmin(np.diff(z_in,axis=1)),4)
        max_z=round(np.nanmax(z_in),4)

        nbins=1+int(round( (max_z-min_z)/min_dz))
        new_z=np.linspace(min_z,max_z,nbins)

        def resamp(orig,axis=-1):
            """ orig: [samples,cells].
            interpolate each sample to from z_in[sample,:] to new_z
            """
            n_samples=orig.shape[0]
            new_A=np.zeros( (n_samples,len(new_z)),np.float64)
            for sample in range(n_samples):
                new_A[sample,:]= np.interp(new_z,
                                           z_in[sample,:],orig[sample,:],
                                           left=np.nan,right=np.nan)
            return new_A

        self.n_ensembles=len(self.ds.sample)
        self.velocity=np.array( (resamp(self.ds.Ve.values),
                                 resamp(self.ds.Vn.values),
                                 resamp(self.ds.Vu.values)) ).transpose(1,2,0)
        self.bin_center_elevation=-new_z # make it negative downward
        self.n_bins=len(new_z)
        self.mtime=utils.to_dnum(self.ds.time.values)
        self.rotation_angle=0
        self.rotation_axes=0
        self.lonlat=np.c_[self.ds.lon.values,self.ds.lat.values]
        self.source=rivr_file
        self.references="Sontek M9"



# a = ADCPSontekM9(rivr_file="040518_BT/040518_7_BTref/20180405125837r.rivr")

transects=[ ADCPSontekM9(rivr_file=f)
            for f in glob.glob("040518_BT/040518_7_BTref/*.rivr")]

for t in transects:
    t.lonlat_to_xy("EPSG:26910")
    
## 
from adcpy import adcpy_recipes

tran_avg=adcpy_recipes.average_transects(transects,dxy=5.0,dz=0.25)

# Seems that this modifies some part of tran_avg in place.
# rot_tran=adcpy_recipes.transect_rotate(tran_avg,rotation='normal')

##
# import pdb
#pdb.run("roz_tran=adcpy_recipes.transect_rotate(tran_avg,rotation='Rozovski')")
roz_tran=adcpy_recipes.transect_rotate(tran_avg,rotation='Rozovski')

##

fig=adcpy.plot.plot_flow_summary(roz_tran,fig=4)

u_ax=fig.axes[1]
u_cax=fig.axes[2]
v_ax=fig.axes[3]
v_cax=fig.axes[4]
u_ax.images[0].set_cmap('seismic')
u_ax.images[0].set_clim([-1,1])

v_ax.images[0].set_cmap('seismic')
v_ax.images[0].set_clim([-0.2,0.2])
