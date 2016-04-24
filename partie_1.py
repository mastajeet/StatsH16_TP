import actuariat as act
import scipy.optimize as optimize
import scipy.stats.distributions as stt
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
print(optimize.minimize(weibull_score.negative_likelyhood_function(),[1,3],method='Nelder-Mead'))
weibull_score.loglikelyhood_function()([1,2])

# d) Construire un itérateur pour executer la méthode newton rhapson
a = weibull_score.loglikelyhood_function()


#Partie 1 : Question 2
import helper as hlp
r_apple = hlp.yf_log_yield_extractor("./data/apple_daily.csv",number_datapoint=150)
r_apple.to_excel('./output/log_yield_apple.xlsx')

student_t = act.random_variable(sample= np.exp(list(r_apple['Yield'][:2])),distribution=stt.t.pdf)
optimisation = optimize.minimize(student_t.optimisable_function,[2,2,2],method='Nelder-Mead')
optimisation.x #degree of freedom, mean, variance