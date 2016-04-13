import numpy as np
import importlib

def simulate_weibull(n, *, alpha, beta):
    # Converti la modélisation du problème à celle de python
    # n = nombre de simulations
    # alpha, beta sont les paramètres de la loi selon le problème
    theta_python = {'a': beta, 'lambda': alpha*np.exp(beta)}
    return theta_python['lambda'] * np.random.weibull(a=theta_python['a'], size=n)

class score_weibull():

    sample = None
    alpha = 0
    beta = 0
    score_alpha = None
    score_beta = None

    def __init__(self,*,sample):
        self.sample = sample
        self.score_weibull()

    def loglikelyhood_function(self):
        return lambda x: ((x[1]-1) * np.sum(np.log(self.sample)) - x[0] * np.sum(np.power(self.sample,x[1])))

    def score_weibull(self):
    # Génère deux fonctions de score pour chacun des paramètres de la weibull
    # Pour la dérivée de la formule, voir document du devoir, formule 1
    # sample est l'échantillon aléatoire

        def score_alpha(sample,alpha,beta):
            return len(sample)/alpha - np.sum(np.power(sample,beta))

        def score_beta(sample,alpha,beta):
            return len(self.sample)/beta + np.sum(np.log(self.sample)) - alpha * np.sum(np.power(sample,beta))

        self.score_alpha = score_alpha
        self.score_beta = score_beta






