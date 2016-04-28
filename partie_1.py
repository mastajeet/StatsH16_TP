from matplotlib.pyplot import hist

import actuariat as act
import matplotlib.pyplot as plt
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

r_apple.sort(columns='Yield')

student_t = act.random_variable(sample=list(r_apple['Yield']), distribution=stt.t.pdf)
optimisation = optimize.minimize(student_t.optimisable_function,[10,np.mean(r_apple['Yield']),np.std(r_apple['Yield'])],method='Nelder-Mead')
parameters_max_vrais = optimisation.x #degree of freedom, mean, variance

# Faire les graphiques
histogramme = plt.hist(r_apple['Yield'], label='Apple Log-Yield',color='g')
line1 = plt.plot(histogramme[1],stt.norm.pdf(histogramme[1],np.mean(r_apple['Yield']),np.std(r_apple['Yield'])),'-x',color='k',label='Normal distribution')
line2 = plt.plot(histogramme[1],stt.t.pdf(histogramme[1],parameters_max_vrais[0],parameters_max_vrais[1],parameters_max_vrais[2]),'-o',color='r',label='Student-T Distribution')
plt.legend(['Normal Distribution','Student-T distribution'])
plt.title('Apple Log-Yield vs distributions')

# commentaire ici!!!
