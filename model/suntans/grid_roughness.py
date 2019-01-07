import stompy.grid.unstructured_grid as ugrid
from stompy.spatial import interp_4d
from stompy.spatial import proj_utils
from stompy.spatial import field
from stompy import utils

import pandas as pd
import numpy as np
##
#g=ugrid.UnstructuredGrid.from_ugrid('snubby-with_bathy.nc')

def add_roughness(g):

    # make a copy of the grid reprojected grid to 1ft dem
    g_ft=g.copy()
    to_ca=proj_utils.mapper('EPSG:26910','EPSG:2227')
    g_ft.nodes['x'][:] = to_ca(g_ft.nodes['x'])
    g_ft.cells_center(refresh=True)

    # for snubby grid it's reasonable to just grab the whole
    # overlapping DEM.
    total_crop=g_ft.bounds()
    # that covers the junction
    dem=field.GdalGrid("../../bathy/dwr/OldRiver_SJ_072020171.tif",
                       geo_bounds=total_crop)
    dem.F[ dem.F<-1e+10 ]=np.nan

    cell_mean=np.zeros(g.Ncells(),np.float64)
    cell_rms=np.zeros(g.Ncells(),np.float64)
    X,Y=dem.XY()

    cell_list=np.arange(g.Ncells()) # eventually may have to process in tiles

    for c in utils.progress(cell_list):
        cell_poly=g_ft.cell_polygon(c)
        try:
            sel=dem.polygon_mask(cell_poly)
        except AttributeError:
            print("!")
            continue
        sel=sel&np.isfinite(dem.F)
        if sel.sum()<10: # ad hoc
            continue

        dem_vals=dem.F[sel]
        dem_x=X[sel]
        dem_y=Y[sel]
        dem_x_loc =dem_x - dem_x.mean()
        dem_y_loc =dem_y - dem_y.mean()

        fit_y = dem_vals
        fit_X = np.c_[dem_x_loc, dem_y_loc, np.ones_like(dem_x)]
        beta_hat = np.linalg.lstsq(fit_X,fit_y,rcond=None)[0]
        flat=np.dot(fit_X,beta_hat)

        anom=dem_vals-flat
        cell_rms[c]=np.sqrt((anom**2).mean())

    # fill in the remaining cells based on averaging neighbors

    df=pd.DataFrame()
    valid=cell_rms!=0.0
    df['value']=cell_rms[valid]
    df['cell']=np.nonzero(valid)[0]
    df['weight']=1.0

    filled=interp_4d.weighted_grid_extrapolation(g,df,cell_col='cell',
                                                 alpha=10)
    g.add_cell_field('bathy_rms',filled,on_exists='overwrite')

if 0:
    plt.figure(1).clf()
    g.plot_edges(lw=0.1,color='k')
    #coll=g.plot_cells(values=cell_rms,cmap='plasma',mask=cell_rms!=0.0)
    coll=g.plot_cells(values=filled,cmap='plasma')
    coll.set_clim([0,0.7])
    plt.axis('equal')
    #plt.axis( (6322822., 6324505, 2116228, 2117796))
    plt.colorbar(coll,label='RMS anomaly (ft)')
