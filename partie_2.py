import actuariat as act
import finance as fin
import scipy.optimize as opt
import inspect
import helper as hlp
import numpy as np


# Section 1 : Actuariat

# a) simuler 100 donnees provenant d'une loi-Poisson chi-carre

q1_number_of_simulation = 100
q1_lambda = 10
q1_alpha = 1/2
q1_beta = 2

##########

experiment = act.poisson_chi_carre()
experiment.set_params(q1_lambda,q1_alpha,q1_beta)

experiment.simulate(q1_number_of_simulation)

# pour obtenir les valeurs de Y
print(experiment.sub_sample_y)

# pour obtenir les valeurs de X, separees par Y
print(experiment.sample)


# b) trouver E[Y] et V[Y] et Q(theta)
print('E[Y]', experiment.expected_value())
print('V[Y]', experiment.variance())

# soient les valeurs theorique des moments

print(inspect.getsource(act.poisson_chi_carre.theoritical_mean))
print(inspect.getsource(act.poisson_chi_carre.theoritical_variance))

# Q(theta)
poisson_chi_carre_quasi_likelyhood = act.quasi_likelyhood(experiment.sub_sample_y)
poisson_chi_carre_quasi_likelyhood.set_mean_function(act.poisson_chi_carre.theoritical_mean)
poisson_chi_carre_quasi_likelyhood.set_variance_function(act.poisson_chi_carre.theoritical_variance)
print(inspect.getsource(poisson_chi_carre_quasi_likelyhood.get_quasi_score))

# c)
optimisation = opt.minimize(poisson_chi_carre_quasi_likelyhood.get_quasi_score_to_minimise,[10,0.5,2],method='Nelder-Mead')
print(optimisation.x)


# d) trouver la prime stoploss
experiment.set_params(optimisation.x[0],optimisation.x[1],optimisation.x[2])
print(experiment.limited_expectation(1))
print(experiment.limited_expectation(2))
print(experiment.limited_expectation(3))
print(experiment.limited_expectation(4))
print(experiment.limited_expectation(5))


#Section 2: Finance

# a)
r_apple = hlp.yf_log_yield_extractor("./data/apple_daily.csv",number_datapoint=150)['Yield']
p_apple = hlp.yf_price_extractor("./data/apple_daily.csv",number_datapoint=150)['Price']
jump_diffusion = act.poisson_jump_diffusion()
jump_diffusion.set_data(r_apple)
jump_diffusion_quasi_likelyhood = act.quasi_likelyhood(sample=jump_diffusion.sub_sample_y)
jump_diffusion_quasi_likelyhood.set_mean_function(act.poisson_jump_diffusion.theoritical_mean)
jump_diffusion_quasi_likelyhood.set_variance_function(act.poisson_jump_diffusion.theoritical_variance)

# b)
mu = np.mean(r_apple)
sigma = np.var(r_apple)

T = 30
mu_projete = mu
sigma_projete = sigma

rf = 30/3650
mu_rn = rf-sigma_projete/T

S0 = p_apple.iloc[0]
K = np.array([115,120,125,130,135])
T = 30

c_black_scholes = fin.black_scholes('C',S0, K, mu_rn, sigma_projete, T)
# array([18.76479768,  14.85631045,  10.94782322,   7.03933599,   3.13084875])

# c)
optimisation = opt.minimize(jump_diffusion_quasi_likelyhood.get_quasi_score_to_minimise,[1,jump_diffusion.expected_value(),jump_diffusion.variance(),0.1,0.1],method='Nelder-Mead')
print(optimisation.x)
# [ 0.02404472 -0.00029647  0.00030939  0.00313     0.15468654]

# d)
x = optimisation.x
jump_diffusion.set_params(x[0],x[1],x[2],x[3],x[4])
h = jump_diffusion.find_h(mu_rn)
# h:-0.28741530702525836

# e)
# pour F tilted, j'obtiens une variance negative, signe qu'il y a un bogue quelque part.
# j'ai eu un probleme similaire lors du calcul de la prime stop-loss ou j'ai du passer par une autre technique...

