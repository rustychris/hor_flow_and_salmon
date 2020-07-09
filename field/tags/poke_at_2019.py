import glob
import os
import pandas as pd
from stompy.spatial import wkb2shp

from shapely import geometry
import numpy as np

## 

utms=glob.glob('2019/UTM/tag*.txt')
##

dfs=[pd.read_csv(fn) for fn in utms]

## 

from shapely import geometry

# Write a shapefile with one feature per tag:
geoms=[]
tags=[]

for df in dfs:
    if len(df)<2: continue
    geo=geometry.LineString( np.c_[ df[' X (East)'],
                                    df[' Y (North)'] ] )
    geoms.append(geo)
    tag=df['Tag ID'].values[0]
    tags.append(tag)
##

from stompy.spatial import wkb2shp
wkb2shp.wkb2shp('tag-lines-2019-utm.shp',geoms,fields=dict(tag=tags),
                overwrite=True)


##

utms=glob.glob('2019/SpeedFilteredUTM/tag*.txt')

dfs=[pd.read_csv(fn) for fn in utms]

# Write a shapefile with one feature per tag:
geoms=[]
tags=[]

for df in dfs:
    if len(df)<2: continue
    geo=geometry.LineString( np.c_[ df[' X (East)'],
                                    df[' Y (North)'] ] )
    geoms.append(geo)
    tag=df['Tag ID'].values[0]
    tags.append(tag)


wkb2shp.wkb2shp('tag-lines-2019-speedfilteredutm.shp',geoms,fields=dict(tag=tags),
                overwrite=True)


