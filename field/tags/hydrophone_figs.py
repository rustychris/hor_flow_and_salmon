# Basic figure showing location of hydrophones

import matplotlib.pyplot as plt
import prepare_pings
from stompy.spatial import field
import stompy.plot.cmap as scmap
from stompy.plot import plot_utils
import six
##

dem=field.GdalGrid("../../bathy/junction-composite-20200604-w_smooth.tif")

##
six.moves.reload_module(prepare_pings)

years=[2018,2019,2020]

plt.figure(1).clf()
fig,axs=plt.subplots(1,len(years),figsize=(6.5,3.75),num=1)

terrain=scmap.load_gradient('BlueYellowRed.cpt')

for ax,year in zip(axs,years):
    rx_locs=prepare_pings.rx_locations(year=year)
    ax.plot(rx_locs.x,rx_locs.y,'go')
    img=dem.plot(ax=ax,cmap=terrain,vmin=-9,vmax=10)

    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)


#plot_utils.scalebar(ax=ax,
