# This is the file used to create the tracks analyzed in
# the swim speed paper

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

# optionally use pre-calculated positions
pre_hydros<-data.table::fread('yap-positions.csv')

serial_blacklist=c()

force_tags<-TRUE
presync<-TRUE

if(FALSE){
  period_dir<-'yaps/full/20180313T1900-20180316T0152'
  fixed_hydros <- NULL # c(5,6) # AM5.0, AM8.8 ?
  time_keeper_idx <- 6 # AM8.8 -- first one to start
}
if(FALSE){
 period_dir<-'yaps/full/20180316T0152-20180321T0003'
 fixed_hydros <- NULL #c(5,6) # AM5.0, AM8.8 ?
 time_keeper_idx <- 6 # AM8.8 -- first one to start
}
# An attempt to run 25 days never finished.
# Running 5 days with just two fixed hydros and 
# bumping the n_offset/day to 8 never finished.
# What if I fix more hydros?
if(FALSE){
  period_dir<-'yaps/full/20180321T0003-20180326T0000'
  fixed_hydros <- NULL # c(5,6) # AM5.0, AM8.8
  time_keeper_idx <- 6 
}

if(FALSE){
    period_dir<-'yaps/full/20180326T0000-20180401T0000'
  fixed_hydros <- NULL # c(5,6)
  time_keeper_idx <- 6
}

if(TRUE){
  # with everybody fixed, the initial go at v04 had
  # some large-ish errors in the sync.
  # only 4 tracks, so maybe not a big deal.
  period_dir<-'yaps/full/20180401T0000-20180406T0000'
  fixed_hydros <-NULL #  c(5,6)
  time_keeper_idx <- 6
}

if(FALSE){
  period_dir<-'yaps/full/20180406T0000-20180410T0000'
  fixed_hydros <- NULL #   c(5,6)
  time_keeper_idx <- 6
}

if(FALSE) {
  # First go, this yielded some really bad sync errors, and just
  # a couple of bad tracks.  Try again.
  # Hydro 7 appears to be the bad guy
  period_dir<-'yaps/full/20180410T0000-20180415T0100'
  serial_blacklist <- c("AM9.0")
  fixed_hydros <- NULL # c(5,6)
  time_keeper_idx <- 6
}

# careful parsing missing sync_tag here
# also, any sync_tag that is not received by at least 3 rxs (self+2)  
# at least once, causes issues. Those have been filtered out upstream.
hydros<-data.table::fread(file.path(period_dir,"hydros.csv"),fill=TRUE,na.strings=c(""))

if(presync) {
  all_detections<-data.table::fread(file.path(period_dir,"all_detections_sync.csv"),fill=TRUE)
} else {
  all_detections<-data.table::fread(file.path(period_dir,"all_detections.csv"),fill=TRUE)
}

# Populate fixed hydros based on what's in pre_hydros
if( is.null(fixed_hydros) ) {
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
}

# Remove detections from blacklisted hydros
hydros <- hydros[ !(hydros$serial %in% serial_blacklist)]
hydros$idx <- 1:nrow(hydros)
all_detections <- all_detections[ !(all_detections$serial %in% serial_blacklist)]
  
# sync_save01: time_keeper=6, fixed=5,6, n_offset_day=2, n_ss_day=2
# results are okay
# v02: same, but n_offset_day=4, n_ss_day=4 
# v03: n_offset_day=8, to match older code.
# v04: try fixing more stations
# v05: bump up ss per day
# v06: presynced data
out_dir<-file.path(period_dir,'v06')
if ( ! dir.exists(out_dir)) {
  dir.create(out_dir)
}


sync_fn=file.path(out_dir,'sync_save')

beacon2018<-c()
beacon2018$hydros <- hydros
beacon2018$detections <- all_detections

# Process tags:
fish_tags<-setdiff( unique(all_detections$tag), hydros$sync_tag)

if ( !presync ) {
  if ( !file.exists(sync_fn) ) {
    # Seems that this is quite sensitive to the "time_keeper_idx"
    inp_sync <- getInpSync(sync_dat=beacon2018, 
                           max_epo_diff = 10,
                           min_hydros = 3, 
                           time_keeper_idx = time_keeper_idx,
                           fixed_hydros_idx = fixed_hydros, # maybe need to avoid SM9, which has no sync tag?
                           n_offset_day = 8,
                           n_ss_day = 8)
    
    sync_model <- getSyncModel(inp_sync,silent=FALSE)
    save(sync_model,file=sync_fn)
    
    plotSyncModelResids(sync_model,by="overall")
    dev.copy(png,file.path(out_dir,"sync-overall.png"))
    dev.off()
    plotSyncModelResids(sync_model,by="sync_tag")
    dev.copy(png,file.path(out_dir,"sync-by_tag.png"))
    dev.off()
    
    plotSyncModelResids(sync_model,by="hydro")
    dev.copy(png,file.path(out_dir,"sync-by_hydro.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="sync_bin_sync")
    dev.copy(png,file.path(out_dir,"sync-bin_sync.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="sync_bin_hydro")
    dev.copy(png,file.path(out_dir,"sync-bin_hydro.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="sync_tag")
    dev.copy(png,file.path(out_dir,"sync-check_by_tag.png"))
    dev.off()
    plotSyncModelCheck(sync_model,by="hydro")
    dev.copy(png,file.path(out_dir,"sync-check_by_hydro.png"))
    dev.off()
  } else {
    load(file=sync_fn)
  }
  
  #NB: this will modify all_detections in place 
  detections_synced <- applySync(toa=all_detections, hydros=hydros, sync_model)
} else {
  # But I need to read in sound speed somewhere, too.
  detections_synced <- all_detections
}

hydros_yaps <- data.table::data.table(sync_model$pl$TRUE_H)
colnames(hydros_yaps) <- c('hx','hy','hz')
max_tries_runYaps <- 5

tag_blacklist=c('7245', '77AB')

for (focal_tag in fish_tags) {
  if ( focal_tag %in% tag_blacklist ) {
    print(paste("BLACKLISTed tag",focal_tag));
    next
  }
  fish_fn<-file.path(out_dir,paste("track-",focal_tag,".csv",sep=""))
  if( (!force_tags) & file.exists(fish_fn)) {
      print(paste("Already processed tag",focal_tag));
    next
  }
  
  print(paste("Processing tag",focal_tag));    
  rbi_min <- 4 # min burst interval
  rbi_max <- 7 # max burst interval

  synced_dat <- detections_synced[tag==focal_tag]
  # on occasion some NA eposync values emerge.  Not sure why.
  invalid <- is.na(synced_dat$eposync)
  if ( sum(invalid)> 0) {
    print(paste(sum(invalid),"NA sync times will be tossed"))
    synced_dat<-synced_dat[!invalid,]
  }
  synced_dat <- synced_dat[order(eposync)]
  
  # early cuts -- 
  if( nrow(synced_dat)<3 ) {
    print(paste("Tag",focal_tag,"has very few observations"))
    next
  }
  # there are other cases where getToaYaps fails, but they seem
  # tied to having very few pings.
  # if no two receivers heard the same ping, first_ping:last_ping
  # generates an error in getToaYaps.
  # if only one ping was heard by multiple receivers, rowMeans causes
  # an error in getToaYaps.
  # Just try it, and bail if it errors.
  toa<-NULL
  tryCatch( { toa <- getToaYaps(synced_dat,hydros_yaps,rbi_min,rbi_max)},
            error=function(e){print(paste("getToaYaps failed for",focal_tag))} )
  if (is.null(toa)){next}
  
  for( try_num in 1:max_tries_runYaps ) {
    # Sort of missing an opportunity to use the precalculated
    # soundspeed.  Would be slightly annoying as the ss is read into synced_dat
    # in long form, but is supplied to getInp with length same as toa.
    inp <- getInp(hydros_yaps, toa, E_dist="Mixture", n_ss=2,pingType='rbi',
                    sdInits=1,rbi_min=rbi_min,rbi_max=rbi_max,ss_data_what="est",
                    ss_data=0)
    yaps_out <- NULL
    tryCatch( { yaps_out <- runYaps(inp,silent=TRUE) },
              error=function(e){print(paste("runYaps failed for ",focal_tag))} )
    if( !is.null(yaps_out) ) { break }
  }
  
  plotYapsEllipses(inp=inp,yaps_out=yaps_out,focal_tag=focal_tag)
  # And save a plot:
  for( i in 1:10 ) {
    img_fn<-file.path(out_dir,paste("track-",focal_tag,"_",i,".png",sep=""))
    if ( !file.exists((img_fn))) {
      break
    }
  }
  png(img_fn, width = 9, height = 7, units = 'in', res = 200)
  plotYapsEllipses(inp=inp,yaps_out=yaps_out,focal_tag=focal_tag)
  dev.off()

  # Write just the real pings to a csv
  pl <- yaps_out$pl
  pl$X <- pl$X + inp$inp_params$Hx0
  pl$Y <- pl$Y + inp$inp_params$Hy0
  pl$top <- pl$top + inp$inp_params$T0
  
  ping_counts<-colSums(yaps_out$rep$mu_toa!=0)  
  real_pings<-(ping_counts>0)
  
  d <- data.frame( x = (yaps_out$pl$X + inp$inp_params$Hx0)[real_pings],
                   y = (yaps_out$pl$Y + inp$inp_params$Hy0)[real_pings],
                   tnum = (yaps_out$pl$top + inp$inp_params$T0)[real_pings],
                   x_sd = yaps_out$plsd$X[real_pings],
                   y_sd = yaps_out$plsd$Y[real_pings],
                   num_rx = ping_counts[real_pings]
                   )
  write.csv(d,fish_fn)
}

