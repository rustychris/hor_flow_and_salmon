"""
Compare gauged flows in/out of the area, to see reversals and 
how close the flows come to closing (i.e. how important is storage,
and how reliable are the flow gauges).
"""
import matplotlib.pyplot as plt
import common
import numpy as np

start=np.datetime64("2018-03-01")
stop =np.datetime64("2018-04-15")

## 
msd=common.msd_flow(start,stop)
sjd=common.sjd_flow(start,stop)
oh1=common.oh1_flow(start,stop)

##

combined=data_comparison.combine_sources([msd.flow_m3s,
                                          sjd.flow_m3s,
                                          oh1.flow_m3s])

net_down=combined.isel(source=1) + combined.isel(source=2)

##

plt.figure(1).clf()

fig,ax=plt.subplots(num=1)

ax.plot(msd.time,msd.flow_m3s,label='MSD')
ax.plot(sjd.time,sjd.flow_m3s,label='SJD')
ax.plot(oh1.time,oh1.flow_m3s,label='OH1')
ax.plot(net_down.time, net_down,label='net down')
ax.legend(loc='upper left')

