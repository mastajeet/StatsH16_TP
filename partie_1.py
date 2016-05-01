import actuariat as act
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.stats.distributions as stt
import numpy as np
import sympy

# Question 1 :
# Analyser les pertes pour la prime
# a) simulater T=20 données d'une loi weibull
theta_al = {'alpha':1,'beta':2}
experiment = act.weibull()
experiment.set_params(theta_al['alpha'],theta_al['beta'])
experiment.simulate(20)

# Reponse
print(experiment.sample)

# b) trouver les vecteur des fonctions scores
# partie theorique

# c) utiliser une fonction d'optimisation pour trouver les estimateurs MV
optimisation = optimize.minimize(experiment.get_score_to_minimise,[1,3],method='Nelder-Mead')
print(optimisation.x)
# {alpha:1.13130645,beta:1.78076148}

# d) Construire un itérateur pour executer la méthode newton rhapson
# ...

# e) calculer l'esperance
beta = sympy.symbols('beta')
alpha = sympy.symbols('alpha')
y = sympy.symbols('y')
x = sympy.symbols('x')

f_x = (alpha*beta*y**(beta-1)*sympy.exp(-alpha*y**beta))
esperance = sympy.integrate(y*f_x,(y,0,sympy.oo))
esperance.subs({alpha:1.13130645,beta:1.78076148})


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
