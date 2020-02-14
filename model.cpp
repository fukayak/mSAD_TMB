#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_INTEGER(I);        // Number of species
    DATA_INTEGER(J);        // Number of geographical units (grid cells)
    DATA_IVECTOR(K);        // Number of plots
    DATA_VECTOR(y);         // Detection-nondetection observations
    DATA_MATRIX(a);         // Area of plots
    DATA_IMATRIX(n_min);    // Auxiliary indices to indicate appropriate data range
    DATA_MATRIX(presence);  // Binary matrix indicating cell-level species presence
    DATA_MATRIX(absence);   // Binary matrix indicating cell-level species absence
    DATA_VECTOR(x1);        // Cell level covariate (AET)
    DATA_VECTOR(x2);        // Cell level covariate (HII)

    PARAMETER(mu);
    PARAMETER(eta);
    PARAMETER_VECTOR(mu_e1);    // mu + e1
    PARAMETER_VECTOR(eta_u1);   // eta + u1
    PARAMETER_VECTOR(e2);
    PARAMETER_VECTOR(e3_input);
    PARAMETER_VECTOR(u2);
    PARAMETER(log_sigma1);
    PARAMETER(log_sigma2);
    PARAMETER(log_sigma3);
    PARAMETER(log_tau1);
    PARAMETER(log_tau2);
    PARAMETER(rho_input);
    PARAMETER(beta1);
    PARAMETER(beta2);
    PARAMETER(beta3);
    PARAMETER(gamma1);
    PARAMETER(gamma2);
    PARAMETER(gamma3);

    using namespace density;
    int count;

    // Correlation coefficient
    Type rho = 2 * (exp(rho_input) / (exp(rho_input) + 1)) - 1;

    // Covariance matrix
    matrix<Type> cov_u2e2(2, 2);
    cov_u2e2(0, 0) = exp(2 * log_tau2);
    cov_u2e2(1, 1) = exp(2 * log_sigma2);
    cov_u2e2(0, 1) = rho * exp(log_tau2) * exp(log_sigma2);
    cov_u2e2(1, 0) = cov_u2e2(0, 1);

    // Array species * cell effect
    matrix<Type> e3(I, J);
    count = 0;
    for (int j = 0; j < J; j++) {
        for (int i = 0; i < I; i++) {
            e3(i, j) = e3_input[count];
            count++;
        }
    }

    // Define log individual density and logit presence probability
    matrix<Type> log_d(I, J);
    matrix<Type> logit_psi(I, J);
    matrix<Type> psi(I, J);
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            log_d(i, j) = mu_e1[i] + beta1 * x1[j] + beta2 * x2[j] + beta3 * x1[j] * x2[j] + e2[j] + e3(i, j);
            logit_psi(i, j) = eta_u1[i] + gamma1 * x1[j] + gamma2 * x2[j] + gamma3 * x1[j] * x2[j] + u2[j];
            psi(i, j) = exp(logit_psi(i, j)) / (exp(logit_psi(i, j)) + 1);
        }
    }

    // Negative log likelihood
    parallel_accumulator<Type> nll(this);

    // Observation component
    for (int j = 0; j < J; j++) {
        if (K[j] == 0) { // For cells without plots
            for (int i = 0; i < I; i++) {
                if (presence(i, j) == 1) {
                    nll -= dbinom(Type(1), Type(1), psi(i, j), 1);
                } else if (absence(i, j) == 1) {
                    nll -= dbinom(Type(0), Type(1), psi(i, j), 1);
                }
            }
        } else { // For cells with plots
            for (int i = 0; i < I; i++) {
                // Define density and detection probability
                vector<Type> d(K[j]);
                vector<Type> p(K[j]);
                for (int k = 0; k < K[j]; k++) {
                    d[k] = exp(log_d(i, j));
                    p[k] = 1 - exp(-a(j, k) * d[k]);
                }

                // Likelihood
                if (presence(i, j) == 1) { // Species for which their cell-level presence is known
                    nll -= dbinom(Type(1), Type(1), psi(i, j), 1);
                    for (int k = 0; k < K[j]; k++) {
                        nll -= dbinom(y[n_min(i, j) + k], Type(1), p[k], 1);
                    }
                } else if (absence(i, j) == 1) { // Species for which their cell-level absense is known
                    nll -= dbinom(Type(0), Type(1), psi(i, j), 1);
                } else { // Other species
                    Type log_prod_1mp = 0;
                    for (int k = 0; k < K[j]; k++) {
                        log_prod_1mp += log(1 - p[k]);
                    }
                    nll -= log(1 - psi(i, j) * (1 - exp(log_prod_1mp)));
                    // This equals to: log(psi[i, j] * prod_k(1 - p[i, j, k]) + (1 - psi[i, j]))
                }
            }
        }
    }

    // Random-effect component
    // Species effect
    for (int i = 0; i < I; i++) {
        nll -= dnorm(mu_e1[i], mu, exp(log_sigma1), 1);
        nll -= dnorm(eta_u1[i], eta, exp(log_tau1), 1);
    }

    // Cell effects (correlated)
    for (int j = 0; j < J; j++) {
        vector<Type> u2e2(2);
        u2e2[0] = u2[j];
        u2e2[1] = e2[j];
        nll += MVNORM(cov_u2e2)(u2e2);
    }

    // Species * cell effects
    for (int m = 0; m < e3_input.size(); m++) {
        nll -= dnorm(e3_input[m], Type(0), exp(log_sigma3), 1);
    }

    // Report
    REPORT(e3);
    REPORT(log_d);
    REPORT(logit_psi);
    REPORT(mu_e1);
    REPORT(eta_u1);
    REPORT(rho);

    Type sigma1 = exp(log_sigma1);
    Type sigma2 = exp(log_sigma2);
    Type sigma3 = exp(log_sigma3);
    Type tau1 = exp(log_tau1);
    Type tau2 = exp(log_tau2);
    ADREPORT(mu);
    ADREPORT(eta);
    ADREPORT(sigma1);
    ADREPORT(sigma2);
    ADREPORT(sigma3);
    ADREPORT(tau1);
    ADREPORT(tau2);
    ADREPORT(rho);
    ADREPORT(beta1);
    ADREPORT(beta2);
    ADREPORT(beta3);
    ADREPORT(gamma1);
    ADREPORT(gamma2);
    ADREPORT(gamma3);

    return nll;
}

/* A robust specification is available:
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_INTEGER(I);        // Number of species
    DATA_INTEGER(J);        // Number of geographical units (grid cells)
    DATA_IVECTOR(K);        // Number of plots
    DATA_VECTOR(y);         // Detection-nondetection observations
    DATA_MATRIX(a);         // Area of plots
    DATA_IMATRIX(n_min);    // Auxiliary indices to indicate appropriate data range
    DATA_MATRIX(presence);  // Binary matrix indicating cell-level species presence
    DATA_MATRIX(absence);   // Binary matrix indicating cell-level species absence
    DATA_VECTOR(x1);        // Cell level covariate (AET)
    DATA_VECTOR(x2);        // Cell level covariate (HII)

    PARAMETER(mu);
    PARAMETER(eta);
    PARAMETER_VECTOR(mu_e1);    // mu + e1
    PARAMETER_VECTOR(eta_u1);   // eta + u1
    PARAMETER_VECTOR(e2);
    PARAMETER_VECTOR(e3_input);
    PARAMETER_VECTOR(u2);
    PARAMETER(log_sigma1);
    PARAMETER(log_sigma2);
    PARAMETER(log_sigma3);
    PARAMETER(log_tau1);
    PARAMETER(log_tau2);
    PARAMETER(rho_input);
    PARAMETER(beta1);
    PARAMETER(beta2);
    PARAMETER(beta3);
    PARAMETER(gamma1);
    PARAMETER(gamma2);
    PARAMETER(gamma3);

    using namespace density;
    int count;

    // Correlation coefficient
    Type rho = 2 * (exp(rho_input) / (exp(rho_input) + 1)) - 1;

    // Covariance matrix
    matrix<Type> cov_u2e2(2, 2);
    cov_u2e2(0, 0) = exp(2 * log_tau2);
    cov_u2e2(1, 1) = exp(2 * log_sigma2);
    cov_u2e2(0, 1) = rho * exp(log_tau2) * exp(log_sigma2);
    cov_u2e2(1, 0) = cov_u2e2(0, 1);

    // Array species * cell effect
    matrix<Type> e3(I, J);
    count = 0;
    for (int j = 0; j < J; j++) {
        for (int i = 0; i < I; i++) {
            e3(i, j) = e3_input[count];
            count++;
        }
    }

    // Define log individual density and logit presence probability
    matrix<Type> log_d(I, J);
    matrix<Type> logit_psi(I, J);
    matrix<Type> log_psi(I, J);
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            log_d(i, j) = mu_e1[i] + beta1 * x1[j] + beta2 * x2[j] + beta3 * x1[j] * x2[j] + e2[j] + e3(i, j);
            logit_psi(i, j) = eta_u1[i] + gamma1 * x1[j] + gamma2 * x2[j] + gamma3 * x1[j] * x2[j] + u2[j];
            log_psi(i, j) = -logspace_add(Type(0), -logit_psi(i, j));
        }
    }

    // Negative log likelihood
    parallel_accumulator<Type> nll(this);

    // Observation
    for (int j = 0; j < J; j++) {
        if (K[j] == 0) { // For cells without plots
            for (int i = 0; i < I; i++) {
                if (presence(i, j) == 1) {
                    nll -= dbinom_robust(Type(1), Type(1), logit_psi(i, j), 1);
                } else if (absence(i, j) == 1) {
                    nll -= dbinom_robust(Type(0), Type(1), logit_psi(i, j), 1);
                }
            }
        } else { // For cells with plots
            for (int i = 0; i < I; i++) {
                // Define density and detection probability
                vector<Type> logit_p(K[j]);
                vector<Type> log_p(K[j]);
                vector<Type> log_1mp(K[j]);
                for (int k = 0; k < K[j]; k++) {
                    log_p[k] = logspace_sub(Type(0), -exp(log(a(j, k)) + log_d(i, j)));
                    log_1mp[k] = -a(j, k) * exp(log_d(i, j));
                    logit_p[k] = log_p[k] - log_1mp[k];
                }

                // Likelihood
                if (presence(i, j) == 1) { // Species for which their cell-level presence is known
                    nll -= dbinom_robust(Type(1), Type(1), logit_psi(i, j), 1);
                    for (int k = 0; k < K[j]; k++) {
                        nll -= dbinom_robust(y[n_min(i, j) + k], Type(1), logit_p[k], 1);
                    }
                } else if (absence(i, j) == 1) { // Species for which their cell-level absense is known
                    nll -= dbinom_robust(Type(0), Type(1), logit_psi(i, j), 1);
                } else { // Other species
                    Type log_prod_1mp = 0;
                    for (int k = 0; k < K[j]; k++) {
                        log_prod_1mp += log_1mp[k];
                    }
                    nll -= logspace_sub(Type(0), log_psi(i, j) + logspace_sub(Type(0), log_prod_1mp));
                    // This equals to: log(psi[i, j] * prod_k(1 - p[i, j, k]) + (1 - psi[i, j]))
                }
            }
        }
    }

    // Random-effect component
    // Species effect
    for (int i = 0; i < I; i++) {
        nll -= dnorm(mu_e1[i], mu, exp(log_sigma1), 1);
        nll -= dnorm(eta_u1[i], eta, exp(log_tau1), 1);
    }

    // Cell effects (correlated)
    for (int j = 0; j < J; j++) {
        vector<Type> u2e2(2);
        u2e2[0] = u2[j];
        u2e2[1] = e2[j];
        nll += MVNORM(cov_u2e2)(u2e2);
    }

    // Species * cell effects
    for (int m = 0; m < e3_input.size(); m++) {
        nll -= dnorm(e3_input[m], Type(0), exp(log_sigma3), 1);
    }

    // Report
    REPORT(e3);
    REPORT(log_d);
    REPORT(logit_psi);
    REPORT(mu_e1);
    REPORT(eta_u1);
    REPORT(rho);

    Type sigma1 = exp(log_sigma1);
    Type sigma2 = exp(log_sigma2);
    Type sigma3 = exp(log_sigma3);
    Type tau1 = exp(log_tau1);
    Type tau2 = exp(log_tau2);
    ADREPORT(mu);
    ADREPORT(eta);
    ADREPORT(sigma1);
    ADREPORT(sigma2);
    ADREPORT(sigma3);
    ADREPORT(tau1);
    ADREPORT(tau2);
    ADREPORT(rho);
    ADREPORT(beta1);
    ADREPORT(beta2);
    ADREPORT(beta3);
    ADREPORT(gamma1);
    ADREPORT(gamma2);
    ADREPORT(gamma3);

    return nll;
}
*/

