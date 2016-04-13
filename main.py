import actuariat as act
import scipy.optimize as optimize
import numpy as np
# Question 1 :
# Analyser les pertes pour la prime
# a) simulater T=20 données d'une loi weibull
theta_al = {'alpha':1,'beta':2}
simulations_weibull = act.simulate_weibull(20, alpha=theta_al['alpha'], beta=theta_al['beta'])

# b) trouver les vecteur des fonctions scores
weibull_score = act.score_weibull(sample=simulations_weibull)
print(weibull_score.score_alpha)
print(weibull_score.score_beta)

# c) utiliser une fonction d'optimisation pour trouver les estimateurs MV
optimize.minimize(weibull_score.loglikelyhood_function(),[1,2],method="TNC",bounds=((0,10),(0,10)))
weibull_score.loglikelyhood_function()([1,2])

# d) Construire un itérateur pour executer la méthode newton rhapson
a = weibull_score.loglikelyhood_function()
