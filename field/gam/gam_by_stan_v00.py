"""
Potentially run stan to better characterize the swim speed results.

Currently the R code has to split out the results into one model 
for the intra-track variation, for which individuals are a random
effect, and a second model for inter-track variation, where summary
stats for tracks are correlated against slow-moving variables.

The other issue I run into with the R/mgcv approach is that it seems
to overestimate significance of results. At least some of this is due
to autocorrelation, but including an AR(1) process does not change the
results much. The calculation of significance is presumably based on the
sample size, and autocorrelation means that there are fewer DOFs in the
data than it appears.

Are there mgcv ways of penalizing the RE term?

How would a Bayesian approach determine significance? have to remind myself
of that part...


"""

##
import numpy as np
import pandas as pd
import pystan
from stompy import utils
import pickle 

## 
def load_model(model_file):
    model_pkl=model_file+'.pkl'
    if utils.is_stale(model_pkl,[model_file]):
        sm = pystan.StanModel(file=model_file)
        with open(model_pkl, 'wb') as fp:
            pickle.dump(sm, fp)
    else:
        with open(model_pkl,'rb') as fp:
            try:
                sm=pickle.load(fp)
            except EOFError:
                os.unlink(model_pkl)
                return load_model(model_file)
    return sm

## 
seg_data = pd.read_csv('../hor_yaps/segment_correlates.csv')

# convert tag id to a factor for use as random effect in mgcv
#seg_data$tag <- factor(seg_data$id)
seg_data['tag'] = seg_data['id'].astype('category')

seg_data['waterdepth'] = seg_data.model_z_eta - seg_data.model_z_bed
seg_data['vor'] = np.abs(seg_data.model_vor_top2m)

cols = ["tag","swim_urel","swim_lat","hydro_speed","turb","hour","flow_100m3s",
        "vor","waterdepth","tnum"]

# mod_data <- start_event(seg_data[cols],"tnum",event="tag")
mod_data = seg_data[cols]

##

data={}
data['Nseg']=len(mod_data)
for col in mod_data.columns:
    data['seg_'+col]=mod_data[col].values

data['seg_tag']=mod_data['tag'].cat.codes

##

# urel ~ hydro_speed
# arrives at negative correlation, seems good.
sm=load_model('model_00.stan')
opt=sm.optimizing(data=data)

# urel ~ hydro_speed + tag(re)
