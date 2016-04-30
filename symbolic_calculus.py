import sympy


# For the question #1 - part Actuariat
# Getting the weibull distribution with Andrew's parameters

beta = sympy.symbols('beta')
alpha = sympy.symbols('alpha')
y = sympy.symbols('y')
x = sympy.symbols('x')


f_x = (alpha*beta*y**(beta-1)*sympy.exp(-alpha*y**beta))
sympy.integrate(f_x,(y,0,x))

# take the output where alpha and beta both are greater than 0 (it takes long)
# -alpha*beta*Piecewise((zoo, And(alpha == 0, beta == 0)), (zoo*exp(-alpha), beta == 0), (0**beta/beta, alpha == 0), (-exp(-0**beta*alpha)/(alpha*beta), True)) + alpha*beta*Piecewise((log(x), And(alpha == 0, beta == 0)), (exp(-alpha)*log(x), beta == 0), (x**beta/beta, alpha == 0), (-exp(-alpha*x**beta)/(alpha*beta), True))
# => -alpha*beta*-exp(-0**beta*alpha)/(alpha*beta) +alpha*beta-exp(-alpha*x**beta)/(alpha*beta)
# little bit of solving :
# => exp(-0**beta*alpha) +-exp(-alpha*x**beta)
# => 1 - exp(-alpha*x**beta)

F_x = 1-sympy.exp(-alpha*x**beta)

u = sympy.symbols('u')
F_x = 1-sympy.exp(-alpha*x**beta)-u

# we solve for y
sympy.solvers.solve(F_x,x)
# Now we have a function we can simulate (0,1) values and generate weibull random variables!
# x = np.sqrt(np.log(-1/(u - 1))





# For the question #2 - part Actuariat,
# getting the derivative of the mgf of a poisson-chi-square random variable

lmbd = sympy.symbols('lmbd')
s = sympy.symbols('s')
beta = sympy.symbols('beta')
alpha = sympy.symbols('alpha')

sympy.diff(sympy.exp(lmbd*((1-beta*s)**(-alpha)-1)),s)
# alpha*beta*lmbd*(-beta*s + 1)**(-alpha)*exp(lmbd*(-1 + (-beta*s + 1)**(-alpha)))/(-beta*s + 1)

# Premier Moment :
sympy.diff(alpha*beta*lmbd*(-beta*s + 1)**(-alpha)*sympy.exp(lmbd*(-1 + (-beta*s + 1)**(-alpha)))/(-beta*s + 1),s).subs(s,0)
# alpha**2*beta**2*lmbd**2 + alpha**2*beta**2*lmbd + alpha*beta**2*lmbd

# Deuxieme Moment :
sympy.diff(alpha*beta*lmbd*(-beta*s + 1)**(-alpha)*sympy.exp(lmbd*(-1 + (-beta*s + 1)**(-alpha)))/(-beta*s + 1),s,2).subs(s,0)
# alpha*beta**3*lmbd*(alpha**2*lmbd**2 + 3*alpha**2*lmbd + alpha**2 + 3*alpha*lmbd + 3*alpha + 2)

# Troisieme Moment :
sympy.diff(-beta*alpha*lmbd*(-beta*s + 1)**alpha*sympy.exp(lmbd*((-beta*s + 1)**alpha - 1))/(-beta*s + 1),s,3).subs(s,0)
# alpha*beta**4*lmbd*(alpha**3*lmbd**3 + 6*alpha**3*lmbd**2 + 7*alpha**3*lmbd + alpha**3 - 6*alpha**2*lmbd**2 - 18*alpha**2*lmbd - 6*alpha**2 + 11*alpha*lmbd + 11*alpha - 6)