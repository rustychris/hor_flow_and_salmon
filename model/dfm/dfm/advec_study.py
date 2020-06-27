import numpy as np
import hor_dfm

base_model=hor_dfm.model

for advec_type in [0,1,2,4,5,3,33]:
    model=base_model.copy()

    model.set_run_dir("runs/hor_005_advec%02d"%advec_type)
    model.mdu['numerics','AdvecType']=advec_type
    # This should be enough time to get to steady state
    model.run_stop=model.run_start + np.timedelta64(4,'h')

    model.write()
    model.partition()
    model.run_model()

