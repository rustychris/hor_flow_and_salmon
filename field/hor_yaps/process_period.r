library(yaps)
library(plotrix)
library(raster)
library(rgdal)

dem_fn=file.path("../../bathy/junction-composite-dem-no_adcp.tif")

dem=raster(dem_fn)
crs(dem) <- '+proj=utm +zone=10 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs' # 'EPSG:26910'
dem_slope <- terrain(dem, opt='slope')
dem_aspect <- terrain(dem, opt='aspect')
hill <- hillShade(dem_slope, dem_aspect, 40, 270)
plot(hill, col=grey(0:100/100), legend=FALSE)


plotYapsEllipses <- function(inp,yaps_out){
  # customized plotting - map view with some indication of uncertainty
  pl <- yaps_out$pl
  plsd <- yaps_out$plsd
  pl$X <- pl$X + inp$inp_params$Hx0
  pl$Y <- pl$Y + inp$inp_params$Hy0
  # pl$top <- pl$top + inp$inp_params$T0
  
  hydros <- data.frame(hx=inp$datTmb$H[,1] + inp$inp_params$Hx0, hy=inp$datTmb$H[,2] + inp$inp_params$Hy0)

  pad<-30  
  xlim=c( min(hydros$hx,pl$X) - pad, max(hydros$hx,pl$X) + pad)
  ylim=c( min(hydros$hy,pl$Y) - pad, max(hydros$hy,pl$Y) +pad)
  
  ext=extent( c(xlim,ylim) ) # 647100,647500,4185400,4186000))
  #plot(dem,ext=ext,col=gray.colors(255))
  plot(hill,ext=ext,col=grey(0:100/100),main=focal_tag)
  
  points(hy~hx, data=hydros, col="green", pch=20, cex=2, asp=1)
  # , xlab="UTM_X", ylab="UTM_Y", main=focal_tag,xlim=xlim,ylim=ylim)
  
  # indicate where there were actual pings, versus just interpolation.
  mu_toa<-yaps_out$rep$mu_toa # [receiver, output_time_index]
  real_pings<-colSums(mu_toa!=0)>0
  
  draw.ellipse(pl$X[real_pings],pl$Y[real_pings],plsd$X[real_pings],plsd$Y[real_pings],border='lightseagreen')
  lines(pl$Y[real_pings]~pl$X[real_pings], col="red")
  points(pl$Y[real_pings]~pl$X[real_pings], col="red",pch=20)
}

# Process a chunk of receiver data as prepared by split_for_yaps.py

# Failed to sync when all hydros fixed.
#period_dir="../circulation/yaps/20180318T0000-20180318T0300"
# mostly good, but 7D49 is wacky.
period_dir="../circulation/yaps/20180318T0000-20180318T0600"

# careful parsing missing sync_tag here
# also, any sync_tag that is not received by at least 3 rxs (self+2)  
# at least once, causes issues. Those have been filtered out upstream.
hydros<-data.table::fread(file.path(period_dir,"hydros.csv"),fill=TRUE,na.strings=c(""))
all_detections<-data.table::fread(file.path(period_dir,"all_detections.csv"),fill=TRUE)

beacon2018<-c()
beacon2018$hydros <- hydros
beacon2018$detections <- all_detections

inp_sync <- getInpSync(sync_dat=beacon2018, max_epo_diff = 10,
                       min_hydros = 2,
                       time_keeper_idx = 2,
                       fixed_hydros_idx = c(1:3),
                       n_offset_day = 4,
                       n_ss_day = 2)

# This seems to be working -- a few outliers, but generally O(0.1m) residuals.
sync_model <- getSyncModel(inp_sync,silent=FALSE)

# seem to be having some trouble while bringing in the rest of the detections.
# running all of the above code, but in a non-fresh environment...
# no different running in fresh environment.
# that was with including 2-rx pings for beacons.  How about going back to forcing
# beacon pings to be received by 3 (in addition to skipping )
# is it b/c I set too many to fixed?
# hmm - 11 and 1 don't even get times for the latter 2 offset intervals.

plotSyncModelResids(sync_model,by="overall")
plotSyncModelResids(sync_model,by="sync_tag")
plotSyncModelResids(sync_model,by="hydro")
plotSyncModelCheck(sync_model,by="sync_bin_sync")
plotSyncModelCheck(sync_model,by="sync_bin_hydro")
plotSyncModelCheck(sync_model,by="sync_tag")
plotSyncModelCheck(sync_model,by="hydro")

detections_synced <- applySync(toa=all_detections, hydros=hydros, sync_model)

hydros_yaps <- data.table::data.table(sync_model$pl$TRUE_H)
colnames(hydros_yaps) <- c('hx','hy','hz')

# Process tags:

fish_tags<-setdiff( unique(all_detections$tag), hydros$sync_tag)

for (focal_tag in fish_tags) {
  print(paste("Processing tag",focal_tag));

  rbi_min <- 4 # min burst interval
  rbi_max <- 7 # max burst interval

  synced_dat <- detections_synced[tag==focal_tag]

  # why rows with all nan?
  toa <- getToaYaps(synced_dat,hydros_yaps,rbi_min,rbi_max)

  inp <- getInp(hydros_yaps, toa, E_dist="Mixture", n_ss=2,pingType='rbi',
                sdInits=1,rbi_min=rbi_min,rbi_max=rbi_max,ss_data_what="est",
                ss_data=0)
  yaps_out <- runYaps(inp,silent=TRUE)
  # gives warnings about NaNs produced?
  # and the track is quite short.
  # what if I give it only triplerx? No warnings, and the track is longer.
  # And when I give it the longer dataset, even though it loses some crap
  # to 11, the overall track is actually longer and looks pretty good.
  plotYapsEllipses(inp=inp,yaps_out=yaps_out)
  dev.copy(png,file.path(period_dir,paste("track-",focal_tag,".png",sep="")))
  dev.off()

  # Write just the real pings to a csv
  pl <- yaps_out$pl
  pl$X <- pl$X + inp$inp_params$Hx0
  pl$Y <- pl$Y + inp$inp_params$Hy0
  pl$top <- pl$top + inp$inp_params$T0
  
  real_pings<-colSums(yaps_out$rep$mu_toa!=0)>0
  
  d <- data.frame( x = (yaps_out$pl$X + inp$inp_params$Hx0)[real_pings],
                   y = (yaps_out$pl$Y + inp$inp_params$Hy0)[real_pings],
                   tnum = (yaps_out$pl$top + inp$inp_params$T0)[real_pings],
                   x_sd = yaps_out$plsd$X[real_pings],
                   y_sd = yaps_out$plsd$Y[real_pings]
                   )
  write.csv(d,file.path(period_dir,paste("track-",focal_tag,".csv",sep="")))
}
