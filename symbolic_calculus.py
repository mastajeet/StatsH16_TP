import sympy


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