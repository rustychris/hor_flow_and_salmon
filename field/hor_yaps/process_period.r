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


plotYapsEllipses <- function(inp,yaps_out,focal_tag=""){
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
  ipad<-300
  ext=extent( c(xlim[1]-ipad,xlim[2]+ipad),
              c(ylim[1]-ipad,ylim[2]+ipad) )
  #plot(dem,ext=ext,col=gray.colors(255))
  plot.new()
  plot.window(xlim=xlim,ylim=ylim,main=focal_tag,asp=1.0)
  title(main=focal_tag)
  plot(hill,legend=FALSE,col=grey(0:100/100),add=TRUE,ext=ext)

  points(hy~hx, data=hydros, col="green", pch=20, cex=2, asp=1)
  # , xlab="UTM_X", ylab="UTM_Y", main=focal_tag,xlim=xlim,ylim=ylim)
  
  # indicate where there were actual pings, versus just interpolation.
  mu_toa<-yaps_out$rep$mu_toa # [receiver, output_time_index]
  real_pings<-colSums(mu_toa!=0)>0
  ping_counts<-colSums(mu_toa!=0)
  
  draw.ellipse(pl$X[real_pings],pl$Y[real_pings],plsd$X[real_pings],plsd$Y[real_pings],border='lightseagreen')
  lines(pl$Y[real_pings]~pl$X[real_pings], col="red")
  # points(pl$Y[real_pings]~pl$X[real_pings], col="red",pch=20)
  
  #cols=c("red","blue","green","yellow","black","cyan")
  cols=brewer.pal(n=max(ping_counts),name='Paired')
  labels=1:max(ping_counts)
  points(pl$Y[real_pings]~pl$X[real_pings], col=cols[ping_counts[real_pings]],pch=20)
  legend(x="topright", legend = labels, col=cols, pch=20,title="# recv")
}



process_period <- function(period_dir,force_tags=FALSE) {
  # careful parsing missing sync_tag here
  # also, any sync_tag that is not received by at least 3 rxs (self+2)  
  # at least once, causes issues. Those have been filtered out upstream.
  hydros<-data.table::fread(file.path(period_dir,"hydros.csv"),fill=TRUE,na.strings=c(""))
  all_detections<-data.table::fread(file.path(period_dir,"all_detections.csv"),fill=TRUE)
  
  # sync_save: original sync, no am3, fixed_hydros==c(4,5)
  # sync_save2: use mean positions from original run, global
  # sync_save3: use positions from original, specific to chunk
  sync_fn=file.path(period_dir,'sync_save3')
  
  if (FALSE) {
    # Read in pre-existing position solutions    
    if ( FALSE ) { 
      # use pre-calculated positions where possible.
      pre_hydros<-data.table::fread('yap-positions.csv')
    } else {
      # DBG
      load(file=file.path(period_dir,'sync_save'))
      pos1 <- data.table::data.table( sync_model$pl$TRUE_H )
      colnames(pos1) <- c('x','y','z')
      pre_hydros<-data.table::data.table()
      pre_hydros$serial <- sync_model$inp_synced$inp_params$hydros$serial
      pre_hydros$yap_x <- pos1$x
      pre_hydros$yap_y <- pos1$y
      pre_hydros$yap_z <- pos1$z
    }
    
    # used to be just c(4,5) all the time.
    fixed_hydros<-c()
    
    for ( idx in 1:nrow(hydros) ) {
      match <- pre_hydros[ pre_hydros$serial == hydros$serial[idx] ]
      if ( nrow(match)==0 ) { next }
      hydros$x[idx] <- match$yap_x
      hydros$y[idx] <- match$yap_y
      hydros$z[idx] <- match$yap_z
      fixed_hydros <- c(fixed_hydros,idx)
    }
  } else {
    fixed_hydros <- c(4,5)
  }
  
  beacon2018<-c()
  beacon2018$hydros <- hydros
  beacon2018$detections <- all_detections

  # Process tags:
  fish_tags<-setdiff( unique(all_detections$tag), hydros$sync_tag)
  if(length(fish_tags)==0) {
    return
  }
  
  # DBG
  if( TRUE ) { # !file.exists(sync_fn) ) {
    # Seems that this is quite sensitive to the "time_keeper_idx"
    inp_sync <- getInpSync(sync_dat=beacon2018, 
                           max_epo_diff = 10,
                           min_hydros = 3, 
                           time_keeper_idx = 4, # was 2. changing this did help.
                           fixed_hydros_idx = fixed_hydros, # maybe need to avoid SM9, which has no sync tag?
                           n_offset_day = 2, # was 4
                           n_ss_day = 2)
    
    sync_model <- getSyncModel(inp_sync,silent=FALSE)
    save(sync_model,file=sync_fn)
    
    plotSyncModelResids(sync_model,by="overall")
    dev.copy(png,file.path(period_dir,"sync-overall.png"))
    dev.off()
    plotSyncModelResids(sync_model,by="sync_tag")
    dev.copy(png,file.path(period_dir,"sync-by_tag.png"))
    dev.off()
    
    plotSyncModelResids(sync_model,by="hydro")
    dev.copy(png,file.path(period_dir,"sync-by_hydro.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="sync_bin_sync")
    dev.copy(png,file.path(period_dir,"sync-bin_sync.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="sync_bin_hydro")
    dev.copy(png,file.path(period_dir,"sync-bin_hydro.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="sync_tag")
    dev.copy(png,file.path(period_dir,"sync-check_by_tag.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="hydro")
    dev.copy(png,file.path(period_dir,"sync-check_by_hydro.png"))
    dev.off()
  } else {
    load(file=sync_fn)
  }

  detections_synced <- applySync(toa=all_detections, hydros=hydros, sync_model)

  hydros_yaps <- data.table::data.table(sync_model$pl$TRUE_H)
  colnames(hydros_yaps) <- c('hx','hy','hz')

  for (focal_tag in fish_tags) {

    fish_fn<-file.path(period_dir,paste("track-",focal_tag,".csv",sep=""))
    if( (!force_tags) & file.exists(fish_fn)) {
      print(paste("Already processed tag",focal_tag));
      next
    }
    
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
    # what if I give it only triplerx?  No warnings, and the track is longer.
    # And when I give it the longer dataset, even though it loses some crap
    # to 11, the overall track is actually longer and looks pretty good.
    plotYapsEllipses(inp=inp,yaps_out=yaps_out,focal_tag=focal_tag)
    # And save a plot:
    for( i in 1:10 ) {
      img_fn<-file.path(period_dir,paste("track-",focal_tag,i,".png",sep=""))
      if ( !file.exists((img_fn))) {
        break
      }
    }
    png(img_fn, width = 9, height = 7, units = 'in', res = 200)
    plotYapsEllipses(inp=inp,yaps_out=yaps_out,focal_tag=focal_tag)
    #dev.copy(png,img_fn) # just dumps the screen version
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
    write.csv(d,fish_fn)
  }
}


# Process a chunk of receiver data as prepared by split_for_yaps.py

if (TRUE) {
  # mostly good, but 7D49 is wacky.
  # period_dir="../circulation/yaps/20180318T0000-20180318T0600"
  # Just trying some other random period:
  # period_dir="../circulation/yaps/20180314T1500-20180314T2100"
  # period_dir="../circulation/yaps/20180327T1100-20180327T1700" 
  # period_dir="../circulation/yaps/20180328T1900-20180329T0100" 
  # period_dir="../circulation/yaps/20180401T2300-20180402T0500"

  #period_dir="../circulation/yaps/20180318T0000-20180318T0300"
  period_dir="../circulation/yaps/20180320T0700-20180320T1300"
  process_period(period_dir,force_tags=TRUE)
} else {
  all_hydros<-Sys.glob("../circulation/yaps/*/hydros.csv")
  for (hydro_fn in all_hydros) {
    period_dir<-dirname(hydro_fn)
    tryCatch( { process_period(period_dir)},
              error=function(e){print(paste("Failed in",period_dir))} )
  }
}

# Hmm - the results are significantly worse for 
# 7652 when using the fixed locations vs. the given 
# locations.

# How different are the two sets of sync_save results?
# And if I load the sync_save original, and use that to seed
# positions of all the receivers, do I still get the uncertainty? 
# load( file=file.path(period_dir,'sync_save'))
# sync_model1<-sync_model
# load( file=file.path(period_dir,'sync_save3') )
# sync_model3<-sync_model
#  
# pos1 <- data.table::data.table( sync_model1$pl$TRUE_H )
# colnames(pos1) <- c('x1','y1','z1')
# pos3 <- data.table::data.table(sync_model3$pl$TRUE_H)
# colnames(pos3) <- c('x3','y3','z3')
#  
# pos<-cbind(pos1,pos3)
# pos$dx <- pos$x3 - pos$x1
# pos$dy <- pos$y3 - pos$y1

# Even when re-using the exact positions, the 7652 solution is
# bogus. -- maybe not. test was not good.

# how different are the two sync models?
# $pl$OFFSET is exactly the same
# SLOPE1, SLOPE2 same except for some 1e-322 differences.
# Speed of sound the same
# TRUE_H the same
# LOG_SIGMA_TOA the same
# LOG_SIGMA_HYDROS_XY is the same...
# TOP is the same...

# Make sure I'm actually seeing different runs.
# Nope.  Somehow fixed_hydros_vec shows c(4,5) for both
