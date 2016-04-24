import actuariat as act
import scipy.optimize as opt
import inspect

# Section 1 : Actuariat

# a) simuler 100 donnees provenant d'une loi-Poisson chi-carre

q1_number_of_simulation = 100
q1_lambda = 10
q1_alpha = 1/2
q1_beta = 2

##########

experiment = act.poisson_chi_carre(q1_lambda,q1_alpha,q1_beta)
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
edgeworth = act.edgeworth_expansion(experiment.sub_sample_y)
edgeworth.F_x(10)

#Section 2: Actuariat

