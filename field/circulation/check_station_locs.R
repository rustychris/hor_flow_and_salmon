library(yaps)
library(plotrix)
library("RColorBrewer")
library(raster)
library(rgdal)

dem_fn=file.path("../../bathy/junction-composite-dem-no_adcp.tif")

dem=raster(dem_fn)
crs(dem) <- '+proj=utm +zone=10 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs' # 'EPSG:26910'
dem_slope <- terrain(dem, opt='slope')
dem_aspect <- terrain(dem, opt='aspect')
hill <- hillShade(dem_slope, dem_aspect, 40, 270)


# Plot locations of the receivers to help figure out which ones should be fixed.
# Would like to hold as many fixed as possible...
#plot(y~x, data=hydros, col="green", pch=20, cex=2, asp=1)
#text(x=hydros$x, y=hydros$y, labels=hydros$serial)

rxs<-NULL
all_syncs<-Sys.glob("../circulation/yaps/*/sync_save")
for (sync_fn in all_syncs) {
  load(file=sync_fn)
  # I think this is the 'solved' locations.
  posns <- data.table::data.table(sync_model$pl$TRUE_H)
  colnames(posns) <- c('yap_x','yap_y','yap_z')
  hydros_inp <- sync_model$inp_synced$inp_params

  posns$serial <- hydros_inp$hydros$serial
  posns$in_x <- hydros_inp$hydros$x
  posns$in_y <- hydros_inp$hydros$y
  posns$in_z <- hydros_inp$hydros$z
  if (is.null(rxs)) {
    rxs<-posns
  } else {
    rxs<-rbind(rxs,posns)
  }
}

# group by serial
# check stddev of yap_x, yap_y
#  stddev of in_x, in_y (presumably 0)
# and how different those are.
rx_mean<-aggregate(rxs,list(serial=rxs$serial),mean)
rx_var <- aggregate(rxs,list(serial=rxs$serial),var)
rx_var <- rx_var[c('serial', 'yap_x','yap_y')]

rx_mean$dx <- rx_mean$yap_x - rx_mean$in_x
rx_mean$dy <- rx_mean$yap_y - rx_mean$in_y

#
pad<-30
xlim<-c(min(rx_mean$yap_x)-pad, max(rx_mean$yap_x)+pad)
ylim<-c(min(rx_mean$yap_y)-pad, max(rx_mean$yap_y)+pad)

ipad<-300
ext=extent( c(xlim[1]-ipad,xlim[2]+ipad),
            c(ylim[1]-ipad,ylim[2]+ipad) )
plot.new()
plot.window(xlim=xlim,ylim=ylim,asp=1.0)
plot(hill,legend=FALSE,col=grey(0:100/100),add=TRUE,ext=ext)

points(yap_y~yap_x, data=rx_mean,col="green", pch=20)
points(in_y~in_x,data=rx_mean,col='blue')

# So they all look decent.
# accept the yap values, write them out
write.csv(rx_mean[ c('serial','yap_x','yap_y','yap_z')], 'yap-positions.csv')
