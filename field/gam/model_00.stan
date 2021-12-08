// drive model (normal dist)
data {
  int<lower=1> Nseg;            // number of segments
  real seg_swim_urel[Nseg];        // long swim velocity
  real seg_swim_lat[Nseg];         // lat swim speed
  real seg_hydro_speed[Nseg];         // lat swim speed
}
parameters {
  real m_hydro; // slope of linear relationship swim_urel ~ hydro
  real inter;  // intercept of linear model
}

model {
  // priors
  m_hydro ~ normal(0,1); // weak prior
  inter ~ normal(0,1); // weak prior

  // likelihood
  for(s in 1:Nseg) {
    seg_swim_urel[s] ~ normal( m_hydro*seg_hydro_speed[s]+inter, 0.1);
  }
}
