//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> N;
  int<lower=1> K;
  vector[N] y;
}

parameters {
  simplex[K] pi;              // mixture weights
  ordered[K] mu;              // ordered means
  real<lower=0.05> sigma;     // shared SD with lower bound for stability
}

model {
  // priors
  pi ~ dirichlet(rep_vector(2.0, K));
  mu ~ normal(4, 1.5);
  sigma ~ lognormal(log(0.6), 0.4);

  // marginalized mixture likelihood
  for (n in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      lps[k] = log(pi[k]) + normal_lpdf(y[n] | mu[k], sigma);
    }
    target += log_sum_exp(lps);
  }
}

generated quantities {
  vector[N] log_lik; # log likelihood for obs m
  vector[N] y_rep;  #posterior predictie value for obs n

  for (n in 1:N) {
    vector[K] lps;
    vector[K] resp;
    real u;
    real cum_prob;
    int z_rep;

    for (k in 1:K) {
      lps[k] = log(pi[k]) + normal_lpdf(y[n] | mu[k], sigma);
    }

    log_lik[n] = log_sum_exp(lps);
    resp = softmax(lps);

    u = uniform_rng(0, 1);
    cum_prob = 0;
    z_rep = 1;

    for (k in 1:K) {
      cum_prob += resp[k];
      if (u <= cum_prob) {
        z_rep = k;
        break;
      }
    }

    y_rep[n] = normal_rng(mu[z_rep], sigma);
  }
}

