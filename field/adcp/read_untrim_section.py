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

try:
    from adcpy import adcpy
except ImportError:
    log.warning("Failed to import adcpy")
    adcpy=None

def parse_section_hydro(filename):
    """
    Parse extract_velocity_section output
    """
    all_sections=[]

    with open(filename,'rt') as fp:
        line=fp.readline().strip()

        while 1: # Loop over transects
            if not line:
                break

            # "section 8B"
            assert line.startswith("section ")
            section_name=line.split(' ')[1]
            line=fp.readline().strip()

            water_columns=[]
            while 1: # Loop over locations within one transect:
                if not line:
                    break

                if line.startswith("section "):
                    break
                # Parse this -- water column data
                # "0.000000 647104.061850 4185777.338411 380 25 25 -9.420000 9.420000"
                dist_m,x,y,idx,k_bed,k_surf,z_bed,z_surf = [float(s)
                                                            for s in line.split()]
                k_bed=int(k_bed) ; k_surf=int(k_surf)

                bins=[]

                for k in range(k_bed,k_surf+1):
                    # each bin "20 -0.108014 0.015937 0.500000"
                    items=fp.readline().strip().split()

                    k=int(items[0])
                    u,v,dz=[float(s) for s in items[1:]]
                    bins.append( dict(k=k,u=u,v=v,dz=dz) )
                water_column=dict(dist_m=dist_m,x=x,y=y,idx=idx,k_bed=k_bed,k_surf=k_surf,
                                  z_bed=z_bed,z_surf=z_surf,bins=bins)
                water_columns.append(water_column)

                line=fp.readline().strip()

            # Close out this section
            all_sections.append( (section_name,water_columns) )

    return all_sections

def parsed_to_ds(section,filename):
    """
    NOTE: As written, this creates a 
    """
    ds=xr.Dataset()
    ds.attrs['name']=section[0]
    ds.attrs['source']="%s:%s"%(filename,section[0])
    ds.attrs['filename']=filename

    profiles=section[1]

    wet_profiles=[ prof for prof in profiles if prof['k_bed']<prof['k_surf'] ]
    k_min=min( [p['k_bed'] for p in wet_profiles] )
    k_max=max( [p['k_surf'] for p in wet_profiles] )

    ds['z_surf']=('sample',), np.array([p['z_surf'] for p in wet_profiles]) # z_surf is positive up
    ds['z_bed'] =('sample',), np.array([-p['z_bed'] for p in wet_profiles]) # p['z_bed'] is positive down

    z_min=ds.z_bed.values.min()
    z_max=ds.z_surf.values.max()

    dz_min=0.1 # could scan, but a sliced surface layer might look really small
    # Resample to evenly spaced vertical axis.  Will be shifting to be positive up, relative to
    # water surface, below.
    z_resamp=np.arange(z_min,z_max+dz_min,dz_min)

    ds['sample']=('sample',), np.arange(len(wet_profiles))
    ds['z']=('cell',),z_resamp # 'cell' is more like 'bin' here.

    dists=np.zeros(len(wet_profiles),'f8')
    Ve=np.nan*np.ones( (len(ds['sample']),len(ds['z'])), 'f8')
    Vn=np.nan*np.ones( (len(ds['sample']),len(ds['z'])), 'f8')
    Vu=np.nan*np.ones( (len(ds['sample']),len(ds['z'])), 'f8')

    for p_i,p in enumerate(wet_profiles):
        # represent discrete cells directly
        # thickness of each layer in the model output
        dzs=[ b['dz'] for b in p['bins'] ]
        # Add a zero thickness at the top to help with going from bins to
        # interfaces
        bin_dz=np.array([0] + dzs)
        # n'th item is the elevation of the bottom of the n'th layer, relative
        # to the surface, with an extra item for the top of the top layer.
        # bin_bottoms=p['z_surf'] - (np.cumsum(bin_dz[::-1])[::-1])
        interface_elevations=(-p['z_bed']) + np.cumsum(bin_dz)

        # should be within text tolerances.
        assert np.abs(interface_elevations[-1] - p['z_surf']) < 0.01

        Ninterface=len(interface_elevations)
        bins=np.searchsorted(interface_elevations,z_resamp)
        valid=(bins>0) & (bins<Ninterface)

        bins=bins.clip(1,Ninterface-1) - 1

        for tgt,src in [ (Ve,'u'),(Vn,'v') ]:
            bin_values=np.array([b[src] for b in p['bins']])
            tgt[p_i]=np.where(valid, bin_values[bins], np.nan)

        Vu[...] = 0*Ve[...] # not reported

    ds['Ve']=('sample','cell'), Ve
    ds['Vn']=('sample','cell'), Vn
    ds['Vu']=('sample','cell'), Vu

    ds['location']=ds['z'] #

    xy=np.array([ [p['x'],p['y']] for p in wet_profiles])
    ds['x_utm']=('sample',),xy[:,0]
    ds['y_utm']=('sample',),xy[:,1]

    ll=proj_utils.mapper('EPSG:26910','WGS84')(xy)
    ds['lon']=('sample',),ll[:,0]
    ds['lat']=('sample',),ll[:,1]
    return ds

if adcpy is not None:
    class ADCPSectionHydro(adcpy.ADCPData):
        """
        Read one section from a section_hydro.txt file
        """
        def __init__(self,ds,**kwargs):
            super(ADCPSectionHydro,self).__init__(**kwargs)

            self.name=ds.attrs['name']
            self.filename=ds.attrs['filename']

            self.ds=ds # section_hydro_to_ds(filename,name)
            self.convert_from_ds()
        def convert_from_ds(self):
            # Set ADCPData members from self.ds

            # use round(...,4) to drop some FP roundoff trash
            # model data comes in with absolute z coordinate, but convert to
            # depth below surface:
            _,z_2d = xr.broadcast(self.ds.Ve, self.ds.z_surf-self.ds.location)
            z_in=z_2d.values # self.ds.location.values
            valid=np.isfinite(self.ds.Ve.values)

            min_z=round(np.nanmin(z_in[valid]),4)
            min_dz=np.round(np.nanmin(np.diff(z_in,axis=1)),4)
            max_z=round(np.nanmax(z_in[valid]),4)

            dz_sgn=np.sign(min_dz)
            min_dz=np.abs(min_dz)

            nbins=1+int(round( (max_z-min_z)/min_dz))
            new_z=np.linspace(min_z,max_z,nbins)

            def resamp(orig,axis=-1):
                """ orig: [samples,cells].
                interpolate each sample to from z_in[sample,:] to new_z
                """
                n_samples=orig.shape[0]
                new_A=np.zeros( (n_samples,len(new_z)),np.float64)
                for sample in range(n_samples):
                    new_A[sample,:]= np.interp(dz_sgn*new_z,
                                               dz_sgn*z_in[sample,:],orig[sample,:],
                                               left=np.nan,right=np.nan)
                return new_A

            self.n_ensembles=len(self.ds.sample)
            self.velocity=np.array( (resamp(self.ds.Ve.values),
                                     resamp(self.ds.Vn.values),
                                     resamp(self.ds.Vu.values)) ).transpose(1,2,0)
            self.bin_center_elevation=-new_z # make it negative downward
            self.n_bins=len(new_z)
            if 'time' in self.ds:
                self.mtime=utils.to_dnum(self.ds.time.values)
            else:
                mtime=utils.to_dnum(np.datetime64("2000-01-01"))
                self.mtime=mtime * np.ones(len(self.ds.sample))

            self.rotation_angle=0
            self.rotation_axes=0
            self.lonlat=np.c_[self.ds.lon.values,self.ds.lat.values]
            self.source=self.ds.attrs['source'] # self.filename
            self.name=self.name
            self.references="UnTRIM"
