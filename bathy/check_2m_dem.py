# Checking the coverage of a 2m xyz file from Ed -
# seems to be just a 2m merging of the 2m DWR datasets.
from stompy.spatial import field

##

fn="/home/rusty/mirrors/ucd-X/UnTRIM/JANET Files/Bay-Delta Bathymetry Data/2m.xyz"
xyz=np.loadtxt(fn,delimiter=",")
fx=field.XYZField(X=xyz[:,:2],F=xyz[:,2])
f=fx.rectify(2.0,2.0)

f.write_gdal('2m.tif')


