data {
  int<lower=1> N;                              
  int<lower=0> N_obs;                           

  array[N_obs] int<lower=1, upper=7> y_obs;     
  array[N_obs] int<lower=1, upper=N> obs_index; 

  array[N] int<lower=0, upper=1> has_lag1;
  array[N] int<lower=0, upper=N> lag1_index;

  array[N] int<lower=0, upper=1> has_lag5;
  array[N] int<lower=0, upper=N> lag5_index;
}

transformed data {
  vector[6] c;
  c[1] = -2.5;
  c[2] = -1.5;
  c[3] = -0.5;
  c[4] =  0.5;
  c[5] =  1.5;
  c[6] =  2.5;
}

parameters {
  real alpha_raw;
  real beta1_raw;
  real beta5_raw;
  real<lower=0> sigma;
  vector[N] z;
}

transformed parameters {
  real alpha;
  real beta1;
  real beta5;
  vector[N] mean_x;
  vector[N] x;

  alpha = alpha_raw;

  // Bound coefficients to a reasonable range
  // 0.45 keeps the recursion from exploding too easily
  beta1 = 0.45 * tanh(beta1_raw);
  beta5 = 0.45 * tanh(beta5_raw);

  for (i in 1:N) {
    mean_x[i] = alpha;

    if (has_lag1[i] == 1)
      mean_x[i] += beta1 * x[lag1_index[i]];

    if (has_lag5[i] == 1)
      mean_x[i] += beta5 * x[lag5_index[i]];

    x[i] = mean_x[i] + sigma * z[i];
  }
}

model {

  alpha_raw ~ normal(0, 1.0);
  beta1_raw ~ normal(0, 1.0);
  beta5_raw ~ normal(0, 1.0);
  sigma ~ exponential(2);   
  z ~ normal(0, 1);

  for (n in 1:N_obs) {
    y_obs[n] ~ ordered_logistic(x[obs_index[n]], c);
  }
}

generated quantities {
  array[N_obs] real log_lik;
  array[N_obs] int<lower=1, upper=7> y_rep_obs;

  real x_new;
  int<lower=1, upper=7> y_new;

  for (n in 1:N_obs) {
    log_lik[n] = ordered_logistic_lpmf(y_obs[n] | x[obs_index[n]], c);
    y_rep_obs[n] = ordered_logistic_rng(x[obs_index[n]], c);
  }

  x_new = normal_rng(alpha, sigma);
  y_new = ordered_logistic_rng(x_new, c);
}
