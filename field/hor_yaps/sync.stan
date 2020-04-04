// STAN model emulating yaps sync approach
// can simplify some, as I'm going to use known
// positions for the hydrophones

data {
  int<lower=1> nh; // number of hydrophones
  int<upper=nh> tk; // which hydrophone is the timekeeper
  int<lower=1> np; // number of beacon pings
  int<lower=1> n_offset_idx; // number of offset periods
  int<lower=1> n_ss_idx; // number of soundspeed periods
  
  real H[nh,3]; // 3-dimensional position of each hydrophone
  real toa_offset[np,nh]; // time-of-arrival for a ping at a hydrophone relative to start of offset period

  int<upper=nh> sync_tag_idx_vec[np]; // which hydro each ping came from.
  
  // DATA_VECTOR(fixed_hydros_vec); // everybody is fixed for the moment.
  
  int<upper=n_offset_idx> offset_idx[np];
  int<upper=n_ss_idx> ss_idx[np];

  real<lower=0> sigma_toa;
}

transformed data {
  real<lower=0> dist_mat[nh,nh];

  real min_toa_offset[np]; // for prior on top

  for(h1 in 1:nh) {
    for(h2 in 1:nh) {
      if(h1 == h2){
        dist_mat[h1, h2] = 0.0;
      } else {
        dist_mat[h1, h2] = sqrt( pow(H[h1,1]-H[h2,1],2) +
                                 pow(H[h1,2]-H[h2,2],2) +
                                 pow(H[h1,3]-H[h2,3],2));
      }
    }
  }

  for(p in 1:np){
    min_toa_offset[p]=100000000;
    for(h in 1:nh){
      if ( !is_nan(toa_offset[p,h]) && (toa_offset[p,h]<min_toa_offset[p])) {
        min_toa_offset[p]=toa_offset[p,h];
      }
    }
  }
}

parameters {
  real top[np]; // when a ping was actually emitted
  real offset[nh,n_offset_idx]; // stepwise time offset per hydro, per period
  // real slope1[nh,n_offset_idx];
  // real slope2[nh,n_offset_idx];
  real<lower=0.0> ss[n_ss_idx];
  // PARAMETER_ARRAY(TRUE_H); // all fixed.

  // real log_sigma_toa;
  // PARAMETER_VECTOR(LOG_SIGMA_HYDROS_XY);
}
transformed parameters {
  // real sigma_toa = exp(log_sigma_toa);
  // vector<Type> SIGMA_HYDROS_XY = exp(LOG_SIGMA_HYDROS_XY);

  real mu_toa[np,nh]; 
  real eps_toa[np,nh];

  // Can I do this here, so that eps_toa is reported?

  // This does trigger the warning about non-linear transform of a parameter

  // What would the jacobian look like?
  // eps_toa[p,h] =toa_offset[p,h] - 
  //               (top[p]
  //                       + dist_mat[sync_tag_idx_vec[p], h]/ss[ss_idx[p]]
  //                       + offset[h, offset_idx[p]]
  //                       + slope1[h, offset_idx[p]]*(toa_offset[p,h]/1E6)
  //                       + slope2[h, offset_idx[p]]*pow((toa_offset[p,h]/1E6),2);
  
  // So when we put a prior on, for example, offset,slope1,slope2, and ss...
  // Start with the prior for ss[i]
  // Hmm - hold off for the moment. It's possible that all of that is just
  // related to priors.  And I'm not using eps_toa in a prior, I'm using it
  // for the log-likelihood.
  
  for(p in 1:np) {
    // iterate hydros in ping p
    for(h in 1:nh) {
      if( !is_nan(toa_offset[p,h])) {
        mu_toa[p,h] = top[p]
          + dist_mat[sync_tag_idx_vec[p], h]/ss[ss_idx[p]]
          + offset[h, offset_idx[p]]
          //+ slope1[h, offset_idx[p]]*(toa_offset[p,h]/1E6)
          // + slope2[h, offset_idx[p]]*pow((toa_offset[p,h]/1E6),2)
          ;
        eps_toa[p,h] = toa_offset[p,h] - mu_toa[p,h];
      } else {
        mu_toa[p,h]=0;
        eps_toa[p,h]=0;
      }
    }
  }
}

model {
  // Prior on offsets
  // OFFSET as 1-D random walks...
  print("A: target=",target());
  
  for(i in 1:n_offset_idx) {
    for(h in 1:nh) {
      if(h==tk) {
        // in yaps the sigmas here are all 0.0000000001
        offset[h,i] ~ normal(0.0,1e-6);
        //slope1[h,i] ~ normal(0.0,1e-6);
        // slope2[h,i] ~ normal(0.0,1e-6);
      } else {
        offset[h,i] ~ normal(0.0,30); // seconds
        // slope1[h,i] ~ normal(0.0,10); // usec/second
        // slope2[h,i] ~ normal(0.0,10); // usec/second**2
      }
    }
  }
  print("B: target=",target());

  //speed of sound component
  for(i in 1:n_ss_idx) {
    ss[i] ~ normal(1450,20);
  }
  print("C: target=",target());

  // prior on time of ping
  for(p in 1:np){
    top[p] ~ normal(min_toa_offset[p],10);
  }
  print("D: target=",target());
  
  // iterate pings
  for(p in 1:np) {
    // iterate hydros in ping p
    for(h in 1:nh) {
      // if ( (!is_nan(rx_t[p,b])) && (!is_nan(rx_t[p,a])) && (a!=b) ) { 
      
      if( !is_nan(toa_offset[p,h])) {
        // yaps: 
        //nll -= log(dt(eps_toa(p,h)/SIGMA_TOA, Type(3.0), false)/SIGMA_TOA);

        // dt:  student t distribution, with dof=3.0
        //                        dof  location scale
        eps_toa[p,h] ~ student_t( 3.0, 0.0,     sigma_toa);
      }
    }
  }
  print("E: target=",target());

}
