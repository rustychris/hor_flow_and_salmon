"""
Simple conversion to read Ed's extract_velocity_sections.py type transect output
into something that adcpy will understand.
"""
import os
import pandas as pd
import numpy as np
import xarray as xr
from stompy.spatial import proj_utils
from stompy import utils,xr_transect

import logging as log

from adcpy import adcpy

class ADCPXrTransectData(adcpy.ADCPData):
    """
    Create an ADCPy instance from an xarray / xr_transect
    dataset.
    """
    def __init__(self,ds,**kwargs):
        super(ADCPXrTransectData,self).__init__(**kwargs)
        self.ds=ds

        if 'name' in ds.attrs:
            self.name=ds.attrs['name']
        else:
            self.name='xr_transect %s'%id(self)
        if 'filename' in ds.attrs:
            self.filename=ds.attrs['filename']
        else:
            self.filename=self.name

        if 'source' in ds.attrs:
            self.source=ds.attrs['source']
        else:
            self.source=self.filename

        self.references="Model"
        self.convert_from_ds()

    def convert_from_ds(self):
        # Set ADCPData members from self.ds
        z_min=float(self.ds.z_int.min())
        z_max=float(self.ds.z_int.max())

        # Always resample -- could be smarter about choosing this value
        min_dz=0.05
        nbins=1+int(round( (z_max-z_min)/min_dz))
        new_z=np.linspace(z_min,z_max,nbins)

        # resampled dataset
        dsr=xr_transect.resample_z(self.ds,new_z)

        # flip to get surface to bed.
        dsr=dsr.isel(layer=slice(None,None,-1))

        # This isn't really optional because ADCPy makes assumptions based
        # on the first coordinate value, in addition to the sign of dz.
        # Maybe need the 0.1 to avoid other ADCPy weirdness?
        # dsr['z_ctr']=0.1 + -dsr['z_ctr'] + dsr['z_ctr'].values[0]
        # That still isn't working
        z=dsr['z_ctr'].values
        z-=z[0] - 0.05 # add the -0.5 to make it clear that we are below water surface
        self.bin_center_elevation=z

        self.n_ensembles=len(dsr.sample)
        self.velocity=np.array( (dsr.Ve.values,
                                 dsr.Vn.values,
                                 dsr.Vu.values) ).transpose(1,2,0)


        self.n_bins=len(new_z)
        if 'time' in self.ds:
            self.mtime=utils.to_dnum(self.ds.time.values)
        else:
            mtime=utils.to_dnum(np.datetime64("2000-01-01"))
            self.mtime=mtime * np.ones(len(self.ds.sample))

        self.rotation_angle=0
        self.rotation_axes=0

        self.lonlat=np.c_[dsr.lon.values,dsr.lat.values]

