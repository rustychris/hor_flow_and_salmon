"""
Created Jun 2018

@author: Ed Gross
@organization: Resource Management Associates
@contact: ed@rmanet.com
@note: extract velocity from cross-sections in UnTRIM
"""

from scipy.spatial.distance import euclidean
from stompy.grid import unstructured_grid as ugrid

import shapely.wkb as wkb
import matplotlib.pyplot as plt
import numpy as np
import shapely
from osgeo import ogr
from scipy.spatial import KDTree
import xarray as xr

grd_file = 'junction_29_w_depth_2007.grd'
grd = ugrid.UnTRIM08Grid(grd_file)
centers = grd.cells_centroid()
kdt =  KDTree(centers)

nc = xr.open_dataset('untrim_hydro_Jan2018.nc')
shp = 'ADCP_sections.shp'
oshp = ogr.Open(shp)

layer = oshp.GetLayer(0)
nlines = layer.GetFeatureCount()
lines = []
name = []
spacing = 5.0 # spacing of output
output_cells = []

fout = 'section_hydro.txt'
fp = open(fout, 'wt')

for i in range(nlines):
    feat = layer.GetNextFeature()
    if not feat:
        break
    geo = feat.GetGeometryRef()
    print "geo",geo
    if geo.GetGeometryName() != 'LINESTRING':
        raise Exception, "All features must be linestrings"
    line = wkb.loads(geo.ExportToWkb())
    if i == 0:
        index = feat.GetFieldIndex("name") # hardwired
    name = feat.GetFieldAsString(index)
    lines.append(line)
    fp.write('section %s\n'%(name))

    for dist in np.arange(0,line.length,spacing):
        xy = line.interpolate(dist)

        # now get cell with nearest cell center
        i = kdt.query(xy)[1]
        #print i, xy
        output_cells.append(i)

        # pull profile at last time step of output for that cell
        kb = nc['Mesh2_face_bottom_layer'][i][-1].values
        kt = nc['Mesh2_face_top_layer'][i][-1].values
        bed_el = nc['Mesh2_face_depth'][i].values
        eta = nc['Mesh2_sea_surface_elevation'][i][-1].values
        #print xy.xy[0][0],xy.xy[1][0],i,centers[i]
        #print dist,xy.xy[0][0],xy.xy[1][0],i,kb,kt,bed_el,eta
        fp.write('%f %f %f %d %d %d %f %f\n'%(dist,xy.xy[0][0],xy.xy[1][0],i,kb,kt,bed_el,eta))
        for k in range(kb,kt):
            u = nc['Mesh2_cell_east_velocity'][i][k][-1]
            v = nc['Mesh2_cell_north_velocity'][i][k][-1]
            vol = nc['Mesh2_face_water_volume'][i][k][-1]
            area = nc['Mesh2_face_wet_area'][i][k][-1]
            if area < 1.e-09:
                dz = 0.0
            else:
                dz = vol/area

            fp.write('%d %f %f %f\n'%(k,u,v,dz))

fp.close()
