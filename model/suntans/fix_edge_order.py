import netCDF4
import numpy as np
from stompy.grid import unstructured_grid
import argparse

parser=argparse.ArgumentParser(description="Check and/or fix face ordering in hydro output")

parser.add_argument( "-u", "--update", help="Update inconsistent cells", action='store_true')
parser.add_argument( "-i", "--input-file", help="Input file", default="ptm_average.nc_0000.nc")
args=parser.parse_args()

nc_in=args.input_file # 'ptm_average.nc_0000.nc'

g=unstructured_grid.UnstructuredGrid.read_ugrid(nc_in)

new_cell_edges=np.zeros_like(g.cells['edges'])
for c in range(g.Ncells()):
    oj=g.cell_to_edges(c,ordered=True,pad=True).clip(-1)
    gj=g.cells['edges'][c].clip(-1)
    gj[gj==999999]=-1 # in case it's suntans average output
    new_cell_edges[c,:]=oj
    
    if np.any(gj!=oj):
        print("Mismatch in cell %d: %s  vs %s"%(c,gj,oj))

if args.update:
    nc=netCDF4.Dataset(nc_in,mode='a')
    v=nc['Mesh2_face_edges']
    v[...]=new_cell_edges
    nc.close()
    
