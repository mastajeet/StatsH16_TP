import numpy as np
import scipy.stats as stt
import numpy.random as randdist

import scipy.optimize as opt
import importlib

class weibull():
    alpha = 0
    beta = 0
    score_alpha = None
    score_beta = None
    sample = np.array([])

    def set_params(self, alpha, beta, *args):
        self.alpha = alpha
        self.beta = beta

    def set_data(self, data):
        self.sample = np.array(list(data))

    def simulate(self, n):
        # On integre la formule via sympy et on resoud pour y
        # on obtient une formule pour laquelle on peut simuler des variables aleatoire
        # uniforme 0,1 pour obtenir des simulations d'une loi weibull!
        # n = nombre de simulations
        # alpha, beta sont les paramètres de la loi selon le problème
        u = randdist.uniform(0,1,n)
        self.sample = (np.log(-1 / (u - 1)) / self.alpha) ** (1 / self.beta)

    def get_score_to_minimise(self, params, *args):
        # params is, in order alpha, beta
        return -np.sum(np.log(
            params[0] * params[1] * self.sample ** (params[1] - 1) * np.exp(-params[0]  * self.sample ** params[1] )
            )
        )



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
    sub_sample_y = np.array([])

    def set_params(self, lmbd, alpha, beta):
        self.lmbd = lmbd
        self.beta = beta
        self.alpha = alpha

    def simulate(self, n):
        poisson = randdist.poisson(self.lmbd, n)
        for nb_variables in poisson:
            sub_sample = randdist.gamma(self.alpha, self.beta, int(nb_variables))
            self.sub_sample_y = np.hstack((self.sub_sample_y, np.sum(sub_sample)))
            self.sample.append(sub_sample)

    def fgm(self):
        return lambda s: np.exp(self.lmda * ((1-self.beta*s)**(-self.alpha)-1))

    def cdf(self,x):
        edgeworth = edgeworth_expansion(self.sub_sample_y)
        return edgeworth.F_x(x,2)

    def limited_expectation(self,x):
        # "G" function moments (based on its moment generating function)
        # Using local variable only to lighten the code
        beta = self.beta
        alpha = self.alpha
        lmbd = self. lmbd

        mean_sub_sample = np.mean(self.sub_sample_y)


        m_1 = alpha**2*(1/beta)**2*lmbd**2 + alpha**2*(1/beta)**2*lmbd + alpha*(1/beta)**2*lmbd/mean_sub_sample
        m_2 = alpha*(1/beta)**3*lmbd*(alpha**2*lmbd**2 + 3*alpha**2*lmbd + alpha**2 + 3*alpha*lmbd + 3*alpha + 2)/mean_sub_sample
        m_3 = alpha*(1/beta)**4*lmbd*(alpha**3*lmbd**3 + 6*alpha**3*lmbd**2 + 7*alpha**3*lmbd + alpha**3 - 6*alpha**2*lmbd**2 - 18*alpha**2*lmbd - 6*alpha**2 + 11*alpha*lmbd + 11*alpha - 6)/mean_sub_sample

        mean = m_1
        variance = m_2 - m_1**2
        skewness = (m_3 - 3*m_1*variance - m_1**3) / (variance**3/2)
        edgeworth = edgeworth_expansion(self.sub_sample_y)
        edgeworth.set_statistics(mean,variance,skewness)

        print('first',edgeworth.F_x(x,1))
        print('second',self.cdf(x))

        return mean_sub_sample*(1-edgeworth.F_x(x,1)) - x*(1-self.cdf(x))

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
        self.set_statistics(np.mean(sample),np.var(sample),stt.skew(sample),stt.kurtosis(sample))

    def get_cumulants(self, x):
        self.C1 = self.skewness / 6
        self.C2 = self.kurtosis / 24
        self.C3 = self.C1 ** 2 / 72
        self.P1 = (1 - x ** 2)
        self.P2 = 3 * x - x ** 3
        self.P3 = 10 * x ** 3 - 15 * x - x ** 5

    def set_statistics(self,mean=0,variance=0,skewness=0,kurtosis=0):
        self.mean = mean
        self.variance = variance
        self.skewness = skewness
        self.kurtosis = kurtosis

    def F_x(self, x, nb_terms = 1):
        x = (x - self.mean) / np.sqrt(self.variance)
        self.get_cumulants(x)
        if nb_terms==1:
            return stt.norm.cdf(x) + (self.C1 * self.P1 * stt.norm.pdf(x)) / np.sqrt(len(self.sample))
        if nb_terms==2:
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
