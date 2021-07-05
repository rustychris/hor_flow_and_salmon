# follow m-clark.github.io intro, use mgcv
library(mgcv)
library(ggplot2)
library(GGally)
library(visreg)
library(dplyr)

# raw data as output by track_analyses.py
seg_data <- read.csv('../hor_yaps/segment_correlates.csv')
# convert tag id to a factor for use as random effect in mgcv
seg_data$tag <- factor(seg_data$id)
seg_data$waterdepth <- seg_data$model_z_eta - seg_data$model_z_bed
seg_data$vor <- abs(seg_data$model_vor_top2m)

cols <- c("tag","swim_urel","swim_lat","hydro_speed","turb","hour","flow_100m3s",
          "vor","waterdepth")
mod_data <- seg_data[cols]

# GAM for instantaneous rheotaxis
# Most of the correlates do not have meaningful variation within a 
# track, for example mean river flow does not change over the course
# of a 20 minute track. Here we lump all between-tag variance in a per-tag
# random effect, and model the remaining within-track variance.
# [mgcv with random effects following notes at
#  https://fromthebottomoftheheap.net/2021/02/02/random-effects-in-gams/ ]

# # Initial model with all potential terms included
# mod_segs <- gam(swim_urel ~ s(hydro_speed,bs="cr") 
#                 + s(vor,bs="cr")
#                 + s(waterdepth,bs="cr")
#                 + s(tag,bs='re'), 
#                 data=mod_data)
# summary(mod_segs)
# visreg(mod_segs)
# concurvity(mod_segs,full=FALSE)
# concurvity(mod_segs,full=TRUE)
# BIC(mod_segs) # -5128

# There is still some conflation between hydro_speed and the tag random
# effect. Would this be more robust if I used the hydro speed anomaly
# instead? Anomaly would either be hydro_speed relative to river velocity,
# or hydro_speed relative to track-averaged hydro speed.
# There is concurvity between vorticity, waterdepth, and hydro_speed.
# Using the observed concurvity estimates, full=FALSE:
#   hydro_speed: 0.05 on vor, 0.30 on waterdepth
#   vor: 0.04 on hydro_speed, 0.06 on waterdepth
#   waterdepth: 0.35 on hydro_speed, 0.21 on vorticity.
# When considering the full concurvity:
#   hydro_speed: 0.90 explained by other terms
#   vor: 0.55 explained by other terms
#   waterdepth: 0.61 explained by other terms.

#mod_segs_vor <- gam(swim_urel ~ + s(vor,bs="cr") + s(tag,bs='re'), 
#                data=mod_data)
#BIC(mod_segs_vor) # -4844. Not as good as mod_segs

#mod_segs_depth <- gam(swim_urel ~ + s(waterdepth,bs="cr") + s(tag,bs='re'), 
#                      data=mod_data)
#BIC(mod_segs_depth) # -4973

mod_segs_hydro <- gam(swim_urel ~ + s(hydro_speed,bs="cr") + s(tag,bs='re'), 
                      data=mod_data)
summary(mod_segs_hydro)
visreg(mod_segs_hydro,'hydro_speed',gg=TRUE) + coord_cartesian(ylim = c(-0.75, 0.75))
visreg(mod_segs_hydro,'tag',gg=TRUE) + coord_cartesian(ylim = c(-0.75, 0.75))
BIC(mod_segs_hydro) # -5054

# Of the segment-based models, mod_segs_hydro is the strongest (by BIC)
# of the single parameter models.
# note that hydro_speed is still concurved with tag.  So tag may be absorbing
# a fair bit of variance related to river flow.

# I do think it would be instructive to remove the per-track hydro_speed mean
# and fit again. Would expect to get a similar GCV and R-sq, but also a much
# lower concurvity. Mostly just to test my understanding of the model

# hydro_by_tag <- mod_data %>% group_by(tag) %>% summarize(mean_hydro = mean(hydro_speed))
# modh_data <- mod_data %>% inner_join(hydro_by_tag)
# modh_data$hydro_anom <- modh_data$hydro_speed - modh_data$mean_hydro
# 
# mod_segs_hydroa <- gam(swim_urel ~ + s(hydro_anom,bs="cr") + s(tag,bs='re'), 
#                        data=modh_data)

# summary(mod_segs_hydroa)
# visreg(mod_segs_hydroa)

# With the per-tag hydro mean removed, observed concurvity becomes 0.127 for hydro_anom
# and 0.015 for tag.
# GCV for the model 0.023378, Devience explained 35.9%, R-sq=0.347.
# These are slightly "better" than mod_segs_hydro, not entirely clear why
# there is a difference. BIC also slightly better.
# concurvity(mod_segs_hydroa)
# BIC(mod_segs_hydroa) # -5089

# Lateral swimming with all potential terms included
# vorticity and hydro speed don't look convincing.
# drop them, and BIC gets more negative. good.

# With no weighting, n=6457
# and all terms are highly significant.
# Changing weights decreased fREML, but did not change
# anything else.
# Gamma makes a difference. It decreases edf from 4.5 and 6
# to 2 and 4. Unfortunately this also decreases the strength
# of the random effects term, decreasing its edf from 105 to 20.
# weights<-rep(0.01,nrow(mod_data))

# Two particular models are of interest for lateral swimming
# This one forgoes a random effect in order to include time of
# day. It bears out the same diurnal signal already discussed
# in the paper.
mod_seglat <- gam(swim_lat ~ s(waterdepth,bs="cr")
                + s(swim_urel,bs="cr")
                + s(hour,bs='cc'),
                knots=list(hour=c(0,24)),
                data=mod_data)
summary(mod_seglat)
visreg(mod_seglat)

mod_seglatre <- gam(swim_lat ~ s(waterdepth,bs="cr") +
                   s(swim_urel,bs="cr") +
                   s(tag,bs='re'),
                  data=mod_data)
summary(mod_seglatre)
visreg(mod_seglatre)

# https://stats.stackexchange.com/questions/289230/selecting-gam-model-link-function-and-autocorrelation-mgcv
# Maybe a better approach for the autocorrelation?
# Unfortunately this fails with an error re non-positive definite,
# maybe suggesting that some tag does not have enough samples?

## 

# Summarize the data per track and fit 
track_data <- seg_data %>% 
              group_by(tag) %>% 
              summarize(swim_lat=mean(swim_lat),
                        urel=mean(swim_urel),
                        swim_lon=mean(abs(swim_urel)),
                        hour=min(hour),
                        turb=mean(turb),
                        speed=mean(hydro_speed),
                        flow=mean(flow_100m3s),
                        reach_velo=mean(reach_velo_ms))

m_lon <- gam(urel ~ s(speed,bs="cr") + s(hour,bs="cc") + s(turb,bs='cr') +
                    s(flow,bs='cr') + s(reach_velo,bs='cr'), 
                data=track_data,knots=list(hour=c(0,24)))
summary(m_lon)
visreg(m_lon)

# hour and flow are near significant at 0.06 and 0.056
# Recall this is for urel!
m_lon_slim <- gam(urel ~ s(hour,bs="cc") + s(flow,bs='cr'), 
             data=track_data,knots=list(hour=c(0,24)))
summary(m_lon_slim)
visreg(m_lon_slim)

#track_data$swim_lon=
  
# this shows hour significant (p<1e-6), and flow (p=0.001), but
# not reach_velo, turb, or mean hydro speed.
# and flow and reach_velo are sort of fighting each other.
# mean hydro speed seems to say nothing. Dropping turb, reach_velo and
# hydro speed, and flow becomes more significant, at p=0.0004.
# however the shape of the flow curve is not very convincing.
# pretty good:
#m_lat <- gam(swim_lat ~ s(hour,bs="cc") + s(flow,bs='cr'), 
#             data=track_data,knots=list(hour=c(0,24)))

# a second term of flow, turb or urel are all signficant on their own 
# and flow+urel is pretty strong.
# seems that turbidity and flow are concurved, with turbidity edged out 
# with gamma around 5, flow becomes uninformative.
m_lat <- gam(swim_lat ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr'), 
             data=track_data,knots=list(hour=c(0,24)),
             gamma=5)
summary(m_lat)
visreg(m_lat,'flow')
# 

# log link function -- doesn't change results much from above.
m_loglat <- gam(log(swim_lat) ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr'), 
             data=track_data,knots=list(hour=c(0,24)),
             gamma=5)
summary(m_loglat)
visreg(m_loglat,scale="response",partial=TRUE)

# Attempting to use a Gamma distribution with log-link in order
# to acknowledge the non-negative nature of swim_lat.
# The coefficients in the fit are now sort of crazy
# The plots make it look like it fits really well, though.  Too well?
# The overall predictions look believable, though.
# Part of this is probably because the link function is inverse, so I
# think I'm seeing the inverse of the true predictors.
# 
m_lat_gamma <- gam(swim_lat ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr',k=3), 
                   data=track_data,knots=list(hour=c(0,24)),
                   family=Gamma)
summary(m_lat_gamma)
# plot on response scale to keep it more physical
# must explicitly request partial plot, 
# https://github.com/pbreheny/visreg/issues/55
# caveat emptor -- residuals may be misleading in
# ways I don't understand
visreg(m_lat_gamma,scale="response",partial=TRUE)

m_lat_invgss <- gam(swim_lat ~ s(hour,bs="cc") + s(urel,bs='cr') + s(flow,bs='cr',k=3), 
                   data=track_data,knots=list(hour=c(0,24)),
                   family=inverse.gaussian)
summary(m_lat_invgss)
visreg(m_lat_invgss,scale="response",partial=TRUE)

# These plots with partial=True I do think are misleading -- something
# is off with the scale of the residuals.




# Full scatter on the result:
fits <- predict(m_lat_gamma,type="response")
data <- track_data
data$fit_swim_lat <- fits

plt <- ggplot(data=data,aes(x=swim_lat,y=fit_swim_lat))+ geom_point()

# r-squared of 0.36 (not too different from the summary R-sq(adj) of 0.32)
cor(fits,track_data$swim_lat)^2

# "Histomancy" as McElreath calls it.  Plot a histogram of swim_lat and stare
# at it to guess the distribution
ggplot(data=data,aes(x=swim_lat)) + 
  geom_histogram() +
  geom_density(alpha=0.2,fill='#8888ee')
# That looks left-skewed

# Better, maybe, is to look at the distribution of residuals
# Not that telling.  Fairly compact distribution around 0.
# We know that there is a lower bound on swimm
ggplot(data=data.frame(res=fits-track_data$swim_lat),
       aes(x=res)) + geom_histogram()
