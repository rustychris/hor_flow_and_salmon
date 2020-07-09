# Look at distributions of vertical position in the Tecno data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stompy.spatial import field
from stompy import memoize
## 
dem=field.GdalGrid("../../bathy/junction-composite-dem.tif")
##

stage_data_url='http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs/B95820/2018/STAGE_15-MINUTE_DATA_DATA.CSV'

@memoize.memoize(lru=10)
def fetch_and_parse(local_file,url,**kwargs):
    if not os.path.exists(local_file):
        d=os.path.dirname(local_file)
        if d and not os.path.exists(d):
            os.makedirs(os.path.dirname(local_file))
        utils.download_url(url,local_file,on_abort='remove')
    return pd.read_csv(local_file,**kwargs)

stage_data=fetch_and_parse('msd-stage.csv',stage_data_url,
                           skiprows=3,parse_dates=['time'],names=['time','stage_ft','quality','notes'])
stage_data['stage_m']=stage_data['stage_ft']*0.3048

# Assume that WDL has data in PST.
stage_data['epoch']=utils.to_unix( stage_data['time'].values+np.timedelta64(8,'h') )

##

tracks=pd.read_csv('cleaned_half meter.csv')

##

tracks['eta']=np.interp( tracks['Epoch_Sec'].values,
                         stage_data['epoch'].values, stage_data['stage_m'])

tracks['z_bed']=dem( np.c_[ tracks.X_UTM.values, tracks.Y_UTM.values] )
tracks['depth']=(tracks['eta']-tracks['z_bed'])
tracks['z_above_bed']=tracks['depth']-tracks['Z_meter']

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
tracks['Z_meter_jitter']=tracks['Z_meter'].values + 0.5*np.random.random(len(tracks))
sns.boxplot( tracks.Node_count.values, -tracks.Z_meter_jitter.values)

##

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
wet=tracks['eta']>tracks['z_bed']
wtracks=tracks[wet]
sns.boxplot( wtracks.Node_count.values, wtracks.z_above_bed)
sns.stripplot( wtracks.Node_count.values, wtracks.z_above_bed)

##

plt.figure(3).clf()
plt.scatter( tracks['X_UTM'], tracks['Y_UTM'], 20, tracks['z_bed'], cmap='jet')
