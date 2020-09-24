import os
import numpy as np

from stompy.spatial import field
from stompy.grid import unstructured_grid
from stompy import memoize
import stompy.plot.cmap as scmap

from matplotlib.path import Path
from matplotlib import cm

here=os.path.dirname(__file__)

@memoize.memoize()
def aerial():
    img=field.GdalGrid(os.path.join(here, "../../../gis/aerial/m_3712114_sw_10_h_20160621_20161004-UTM-crop.tif"))
    # Alpha channel is mis-scaled.  Drop it.
    img.F=img.F[...,:3]
    return img

fade_plasma=scmap.transform_color(lambda rgb: 0.5*rgb+0.5, cm.plasma)

def composite_aerial_and_contours(ax):                       
    # faded aerial
    img=aerial().plot(ax=ax,alpha=0.4,zorder=-2)

    cset=masked_dem().contourf(np.linspace(-8,4,8),cmap=fade_plasma,ax=ax,zorder=-1,
                               extend='both')
    return img,cset

def base_grid():
    return unstructured_grid.UnstructuredGrid.read_ugrid(os.path.join(here, "../../../model/grid/snubby_junction/snubby-08-edit60.nc"))

@memoize.memoize()
def grid_poly():
    return base_grid().boundary_polygon()

def dem():
    return field.GdalGrid(os.path.join(here,"../../../bathy/junction-composite-20200604-w_smooth.tif"))

@memoize.memoize()
def demc():
    return dem().crop([647000,647550,4.185550e6,4.186070e6])
    
def masked_dem():
    poly=grid_poly()
    pnts=np.array(poly.exterior)
    p=Path(pnts)

    mask=demc().polygon_mask(poly)
    mask_demc=demc().copy()
    mask_demc.F[~mask]=np.nan
    return mask_demc
