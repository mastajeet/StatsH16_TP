import numpy as np
import scipy.stats as stt
import numpy.random as randdist

import scipy.optimize as opt
import importlib


def weibull_distribution_pdf(x, alpha, beta):
    return alpha * beta * x ** (beta - 1) * np.exp(-alpha * x ** beta)


def simulate_weibull(n, *, alpha, beta):
    # Converti la modélisation du problème à celle de python
    # n = nombre de simulations
    # alpha, beta sont les paramètres de la loi selon le problème
    theta_python = {'a': beta, 'lambda': alpha * np.exp(beta)}
    return theta_python['lambda'] * np.random.weibull(a=theta_python['a'], size=n)


class score_weibull():
    sample = None
    alpha = 0
    beta = 0
    score_alpha = None
    score_beta = None

    def __init__(self, *, sample):
        self.sample = sample
        self.score_weibull()

    def loglikelyhood_function(self):
        return lambda x: -1 * ((x[1] - 1) * np.sum(np.log(self.sample)) - x[0] * np.sum(np.power(self.sample, x[1])))

    def score_weibull(self):
        # Génère deux fonctions de score pour chacun des paramètres de la weibull
        # Pour la dérivée de la formule, voir document du devoir, formule 1
        # sample est l'échantillon aléatoire

        def score_alpha(sample, alpha, beta):
            return len(sample) / alpha - np.sum(np.power(sample, beta))

        def score_beta(sample, alpha, beta):
            return len(self.sample) / beta + np.sum(np.log(self.sample)) - alpha * np.sum(np.power(sample, beta))

        self.score_alpha = score_alpha
        self.score_beta = score_beta


class random_variable():
    sample = []
    distribution = None

    def __init__(self, *, sample, distribution):
        self.sample = sample
        self.distribution = distribution

    def optimisable_function(self, param):
        return self.log_likelyhood_score(param[0], param[1], param[2])

    def log_likelyhood_score(self, df, loc, scale):
        return -1 * np.sum(np.log(self.distribution(self.sample, df, loc, scale)))


class poisson_chi_carre:
    lmbda = 0
    gamma = 0
    beta = 0
    sample = []
    sub_sample_y = []

    def __init__(self):
        self.sample = []
        self.sub_sample_y = np.array([])

    def set_params(self, lmbd, alpha, beta):
        self.lmbd = lmbd
        self.alpha = alpha
        self.beta = beta

    def simulate(self, n):
        poisson = randdist.poisson(self.lmbd, n)
        for nb_variables in poisson:
            sub_sample = randdist.gamma(self.alpha, self.beta, int(nb_variables))
            self.sub_sample_y = np.hstack((self.sub_sample_y, np.sum(sub_sample)))
            self.sample.append(sub_sample)

    def set_data(self,data):
        self.sub_sample_y = np.array(list(data))

    def expected_value(self):
        return np.mean(self.sub_sample_y)

    def variance(self):
        return np.var(self.sub_sample_y)

    def theoritical_mean(params):
        # in order: lambda, alpha, beta
        return params[0] * params[1] * params[2]

    def theoritical_variance(params):
        # in order: lambda, alpha, beta
        return params[0] * (params[1] * params[2] ** 2 + (params[1] * params[2]) ** 2)


class edgeworth_expansion:
    sample = []

    def __init__(self, sample):
        self.sample = sample

    def get_cumulants(self, x):
        self.C1 = stt.skew(self.sample) / 6
        self.C2 = stt.kurtosis(self.sample) / 24
        self.C3 = self.C1 ** 2 / 72
        self.P1 = (1 - x ** 2)
        self.P2 = 3 * x - x ** 3
        self.P3 = 10 * x ** 3 - 15 * x - x ** 5

    def F_x(self, x):
        x = np.sqrt(len(self.sample)) * (x - np.mean(self.sample)) / np.var(self.sample)
        self.get_cumulants(x)
        return stt.norm.cdf(x) + (self.C1 * self.P1 * stt.norm.pdf(x)) / np.sqrt(len(self.sample)) + (
                                                                                                     self.C2 * self.P2 + self.C3 * self.P3) / len(
            self.sample) * stt.norm.pdf(x)


class quasi_likelyhood:
    sample = []
    quasi_mean = None
    quasi_variance = None

    def __init__(self, sample):
        self.sample = sample

    def set_mean_function(self, func):
        self.quasi_mean = func

    def set_variance_function(self, func):
        self.quasi_variance = func

    def get_quasi_score(self, params):
        return np.sum(np.log(
            1 / (2 * 3.141592654) * 1 / (np.sqrt(self.quasi_variance(params))) * np.exp(
                -1 * (self.sample - self.quasi_mean(params)) ** 2 / (self.quasi_variance(params)))
        ))

    def get_quasi_score_to_minimise(self, params):
        return -np.sum(np.log(
            1 / (2 * 3.141592654) * 1 / (np.sqrt(self.quasi_variance(params))) * np.exp(
                -1 * (self.sample - self.quasi_mean(params)) ** 2 / (self.quasi_variance(params)))
        ))

class poisson_jump_diffusion:
    sample = []
    lmbda = 0

    mu_jump = 0
    sigma_jump = 0

    mu_yield = 0
    sigma_yield = 0

    sub_sample_y = []
    sub_sample_mean = []
    sub_sample_variance = []

    def __init__(self):

        self.sub_sample_mean = np.array([])
        self.sub_sample_variance = np.array([])
        self.sub_sample_y = np.array([])

    def set_params(self, lmbd, mu_jump, sigma_jump, mu_yield, sigma_yield):
        self.lmbda = lmbd
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.mu_yield = mu_yield
        self.sigma_yield = sigma_yield

    def simulate(self, n):
        poisson = randdist.poisson(self.lmbd, n)
        for nb_variables in poisson:
            baseline_yield = randdist.normal(self.mu_yield, self.sigma_yield, 1)
            sub_sample = randdist.normal(self.mu_jump, self.sigma_jump, int(nb_variables))
            self.sub_sample_y = np.hstack((self.sub_sample_y, np.sum(sub_sample)+baseline_yield))
            self.sample.append(np.array([baseline_yield,sub_sample]))

    def set_data(self,data):
        self.sub_sample_y = np.array(list(data))

    def expected_value(self):
        return np.mean(self.sub_sample_y)

    def variance(self):
        return np.var(self.sub_sample_y)

    def theoritical_mean(params):
        # in order: lambda, mu_yield, sigma_yield, mu_jump, sigma_jump
        return params[1] + params[0] * params[3]

    def theoritical_variance(params):
        # in order: lambda, alpha, beta
        return params[2]**2 + params[0] * (params[3]**2+params[4]**2)
