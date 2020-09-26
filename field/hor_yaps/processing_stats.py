"""
Generate some summary numbers from the ping processing for
the swimspeed manuscript
"""
import os
import pandas as pd
from shapely import geometry
from stompy.grid import unstructured_grid
from stompy.plot import plot_wkb

import glob
import six
## 
# From sync_pings
data_dirs=[
        'yaps/full/20180313T1900-20180316T0152',
        'yaps/full/20180316T0152-20180321T0003',
        'yaps/full/20180321T0003-20180326T0000',
        'yaps/full/20180326T0000-20180401T0000',
        'yaps/full/20180401T0000-20180406T0000',
        'yaps/full/20180406T0000-20180410T0000',
        'yaps/full/20180410T0000-20180415T0100'
]

dfs=[ pd.read_csv(os.path.join(data_dir,'all_detections.csv'))
      for data_dir in data_dirs]

##

df_tot=pd.concat(dfs)

print("Total detections (non-multipath, clipped to time in the water): ",len(df_tot))
# Total detections:  2765610
# That's about half the number from just counting lines in DET files, with
# the discrepancy due mostly to out-of-water pings, but also some multipath.

## 

# Remove sync pings:
hydros=pd.read_csv("../circulation/hydros.csv")
hydro_tags=hydros['sync_tag'].values

##

df_nonhydro=df_tot[ ~df_tot.tag.isin(hydro_tags).values ]

print(f"Detections from non-sync tags: {len(df_nonhydro)}")
# Detections from non-sync tags: 183951

##

# And further limit to known fish tags.
fish_tags_df=pd.read_excel("../circulation/2018_Data/2018FriantTaggingCombined.xlsx")
fish_tags=fish_tags_df['TagID_Hex'].values

df_fish=df_nonhydro[ df_nonhydro.tag.isin(fish_tags) ]

print(f"Detections from known fish tags: {len(df_fish)}")
# Detections from known fish tags: 147991

print("Unique fish tags received: ",len(df_fish.tag.unique()))
# Unique fish tags received:  346

##

g=unstructured_grid.UnstructuredGrid.read_ugrid("../../model/grid/snubby_junction/snubby-08-edit60.nc")
g_poly=g.boundary_polygon()

# comparison to tekno solutions:
tekno=pd.read_csv("../tags/cleaned_half meter.csv")




# 'raw' output from YAPS;
raw_csv_patt='yaps/full/v06/track*.csv'
raw_csvs=glob.glob(raw_csv_patt)

dfs=[]
for fn in raw_csvs:
    df=pd.read_csv(fn)
    del df['Unnamed: 0']
    basename=os.path.basename(fn)
    df['id']=basename.replace('track-','').replace('.csv','')
    dfs.append(df)
    
yaps=pd.concat(dfs)
yaps3=yaps[ yaps.num_rx>=3 ].copy()

yaps3['in_poly']=[ g_poly.contains( geometry.Point(r.x,r.y) )
                   for idx,r in yaps3.iterrows() ]

tekno['in_poly']=[ g_poly.contains( geometry.Point(r.X_UTM,r.Y_UTM) )
                   for idx,r in tekno.iterrows() ]


## 
# Trim to common periods:
t_min=max( tekno.Epoch_Sec.min(),
           yaps3.tnum.min() )
t_max=min( tekno.Epoch_Sec.max(),
           yaps3.tnum.max() )

yaps3_clip=yaps3[ (yaps3.tnum>=t_min) & (yaps3.tnum<=t_max) ]
tekno_clip=tekno[ (tekno.Epoch_Sec>=t_min) & (tekno.Epoch_Sec<=t_max) ]

##

yaps_in=yaps3_clip.in_poly.sum()
yaps_tot=len(yaps3_clip)
print(f"{yaps_in}/{yaps_tot} ({100.0*yaps_in/yaps_tot:.2f}% 3+ rx yaps solutions are within the domain")
print(f"{yaps_tot-yaps_in}/{yaps_tot} ({100.0*(1.0-yaps_in/yaps_tot):.2f}% 3+ rx yaps solutions are outside the domain")
tek_in=tekno_clip.in_poly.sum()
tek_tot=len(tekno_clip)
print(f"{tek_in}/{tek_tot} ({100.0*tek_in/tek_tot:.2f}% Teknologic solutions are within the domain")
print(f"{tek_tot-tek_in}/{tek_tot} ({100.0*(1.0-tek_in/tek_tot):.2f}% Teknologic solutions are outside the domain")

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

sel=tekno_clip.in_poly
ax.plot(tekno_clip.loc[sel,'X_UTM'],tekno_clip.loc[sel,'Y_UTM'],'k.')
sel=~sel
ax.plot(tekno_clip.loc[sel,'X_UTM'],tekno_clip.loc[sel,'Y_UTM'],'k+')

sel=yaps3_clip.in_poly
ax.plot(yaps3_clip.loc[sel,'x'],yaps3_clip.loc[sel,'y'],'r.')
sel=~sel
ax.plot(yaps3_clip.loc[sel,'x'],yaps3_clip.loc[sel,'y'],'r+')

plot_wkb.plot_wkb(g_poly,ax=ax,zorder=-2,alpha=0.3)
ax.axis('equal')
