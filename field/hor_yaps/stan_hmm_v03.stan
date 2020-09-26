// v02: include a heading term as explanatory variable
// v03: re-enable the other terms
data {
  int<lower=1> K;               // number of states (1 = holding, 2 = active)
  int<lower=1> N;               // length of process = number of segments
  real u[N];                    // swim speed
  real v[N];                    // swim heading turn angle
  real hdg[N];                  // ground heading relative to hydro
  matrix<lower=0>[K,K] alpha;   // transit prior
  vector<lower=0>[K] spd_beta;  // sd u
}
parameters {
  simplex[K] theta[K];          // transit probs
  // enforce an ordering: phi[1] <= phi[2]
  // original example had an ordering for both phi and lambda.
  // have to be sure that each parameter is similarly increasing then
  // this is the shape parameter for a gamma.
  vector<lower=0>[K] spd_alpha;      // emission parameter for swim speed (gamma shape for speed)
  
  // Apparently it can't be both ordered and constrained, as the two imply different transforms
  // the ordering is more of an issue once sampling, while the constraints are required
  // to stay within the support of functions
  vector<lower=0>[K] lambda;   // emission parameter for turn    (concentration in turn angle dist)

  // Parameters for heading 
  // 
  vector[K] hdg_vecx;
  // hoping that this will distinguish the states reliably
  ordered[K] hdg_vecy;
}

transformed parameters {
  vector[K] hdg_conc;
  vector[K] hdg_mean;
  for (k in 1:K) {
    hdg_conc[k]=sqrt(pow(hdg_vecx[k],2) + pow(hdg_vecy[k],2));
    if( hdg_conc[k] < 1e-5 ) {
      hdg_mean[k]=0;
    } else {
      hdg_mean[k]=atan2(hdg_vecy[k],
                        hdg_vecx[k]);
    }
  }
}
model {
  real tmp;
  // priors
  for (k in 1:K) {
    target += dirichlet_lpdf(theta[k] | alpha[k,]'); // ');
    // pretty arbitrary -- at least get some spread
    // try to get y values above/below 0
    tmp = normal_lpdf( hdg_vecx[k] | 10, 20)
      + normal_lpdf( hdg_vecy[k] | -5 + 10*(k-1.0)/(K-1.0), 20);
    target += tmp;

    // for now give up on trying to be specific about which states are which
    // gamma mean is alpha/beta.  here beta is 0.5,
    target+= gamma_lpdf(spd_alpha[k] | spd_beta*0.10, 0.2);
    target+= exponential_lpdf(lambda[k] | 0.1);
  }

  // forward algorithm
  {
    real acc[K];
    real gamma[N,K];
    for (k in 1:K) {
      gamma[1,k]=0.0;
      gamma[1,k] += gamma_lpdf(u[1] | spd_alpha[k], spd_beta[k]);
      // if (lambda[k] < 100)  // Funny business to work around numerical instability of vM
      //   gamma[1,k] += von_mises_lpdf(v[1] | 0, lambda[k]);
      // else
      //   gamma[1,k] += normal_lpdf(v[1] | 0, sqrt(1 / lambda[k]));

      gamma[1,k] += von_mises_lpdf(hdg[1] | hdg_mean[k], hdg_conc[k]);
    }
    
    for (t in 2:N) { // time step
      for (k in 1:K) { // transition to state k
        for (j in 1:K) { // from state j
          acc[j]= gamma[t-1,j] + log(theta[j,k]);
          acc[k]+= gamma_lpdf(u[t] | spd_alpha[k], spd_beta[k]);
          
          // if ( lambda[k]<100)
          //   acc[j] += von_mises_lpdf(v[t] | 0, lambda[k]);
          // else
          //   acc[j] += normal_lpdf(v[t] | 0, sqrt(1/lambda[k]));
          
          acc[j] += von_mises_lpdf(hdg[t] | hdg_mean[k],hdg_conc[k]);
        }
        gamma[t,k] = log_sum_exp(acc);
      }
    }
    target+= log_sum_exp(gamma[N]);
  }
}

generated quantities {
  int<lower=1,upper=K> z_star[N];
  real log_p_z_star;
  // Viterbi algorithm
  {
    int back_ptr[N,K];
    real best_logp[N,K];
    for (k in 1:K) {
      // This had been using K, but shouldn't it be k?
      best_logp[1,k] = 0.0;
      
      best_logp[1,k] += gamma_lpdf(u[1] | spd_alpha[k], spd_beta[k]);
      
      // if ( lambda[k]<100 )
      //   best_logp[1,k] += von_mises_lpdf(v[1] | 0,lambda[k]);
      // else
      //   best_logp[1,k] += normal_lpdf(v[1] | 0,sqrt(1/lambda[k]) );
      best_logp[1,k] += von_mises_lpdf(hdg[1] | hdg_mean[k], hdg_conc[k]);
    }
    for (t in 2:N) {
      for (k in 1:K) {
        best_logp[t,k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          logp = best_logp[t-1,j] + log(theta[j,k]);
          logp += gamma_lpdf(u[t] | spd_alpha[k], spd_beta[k]);
          // if (lambda[k]<100 )
          //   logp += von_mises_lpdf(v[t] | 0, lambda[k]);
          // else
          //   logp += normal_lpdf(v[t] | 0, sqrt(1/lambda[k]));

          logp += von_mises_lpdf(hdg[t] | hdg_mean[k], hdg_conc[k]);
          
          if (logp > best_logp[t,k]) {
            back_ptr[t,k] = j;
            best_logp[t,k] = logp;
          }
        }
      }
    }
    log_p_z_star = max(best_logp[N]);
    for (k in 1:K)
      if (best_logp[N,k] == log_p_z_star)
        z_star[N] = k;
    for (t in 1:(N - 1))
      z_star[N - t] = back_ptr[N - t + 1, z_star[N - t + 1]];
  }
}
