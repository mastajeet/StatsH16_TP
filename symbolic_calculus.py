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



# For the question #2 - part Finance
# Solving for h in the esscher transformation for risk neutral equation


mu_yield = sympy.symbols('mu_yield')
sigma_yield = sympy.symbols('sigma_yield')
mu_jump = sympy.symbols('mu_jump')
sigma_jump = sympy.symbols('sigma_jump')
lmbda = sympy.symbols('lmbda')
h = sympy.symbols('h')
r = sympy.symbols('r')

normal_mgf_exponential_part_yield_h1 = mu_yield*(1+h)+1/2*sigma_yield**2*(1+h)**2
normal_mgf_exponential_part_jump_h1 = lmbda*(mu_jump*(1+h)+1/2*sigma_jump**2*(1+h)**2)

normal_mgf_exponential_part_yield_h = mu_yield*(h)+1/2*sigma_yield**2*(h)**2
normal_mgf_exponential_part_jump_h = lmbda*(mu_jump*(h)+1/2*sigma_jump**2*(h)**2)


sympy.solvers.solve(normal_mgf_exponential_part_yield_h1+normal_mgf_exponential_part_jump_h1 - normal_mgf_exponential_part_yield_h - normal_mgf_exponential_part_jump_h - r,h)


# now creating the 3 moments for F tilted by h, and G, F tilted by h+1

S0 = sympy.symbols('S0')
s = sympy.symbols('s')
T = sympy.symbols('T')


mu_yield = sympy.symbols('mu_yield')
sigma_yield = sympy.symbols('sigma_yield')
mu_jump = sympy.symbols('mu_jump')
sigma_jump = sympy.symbols('sigma_jump')
lmbda = sympy.symbols('lmbda')
h = sympy.symbols('h')
r = sympy.symbols('r')

normal_mgf_exponential_part_yield_h1 = mu_yield*(s+h)+1/2*sigma_yield**2*(s+h)**2
normal_mgf_exponential_part_jump_h1 = lmbda*(mu_jump*(s+h)+1/2*sigma_jump**2*(s+h)**2)

normal_mgf_exponential_part_yield_h = mu_yield*(h)+1/2*sigma_yield**2*(h)**2
normal_mgf_exponential_part_jump_h = lmbda*(mu_jump*(h)+1/2*sigma_jump**2*(h)**2)

MfT = S0*sympy.exp(s)*((sympy.exp(normal_mgf_exponential_part_yield_h1)*sympy.exp(normal_mgf_exponential_part_jump_h1))/(sympy.exp(normal_mgf_exponential_part_yield_h)*sympy.exp(normal_mgf_exponential_part_jump_h)))**T

m_1 = sympy.diff(MfT,s)
m_2 = sympy.diff(MfT,s,2)
m_3 = sympy.diff(MfT,s,3)

# for purpose of question

def part2_finance_substitution(sympy_formula):
    return sympy_formula.subs({s:0,T:30,S0:108.66,T:30,lmbda:0.02404472,mu_yield:-0.00029647,sigma_yield:0.00030939,mu_jump:0.00313,sigma_jump:0.15468654,h:-0.28741530702525836})

part2_finance_substitution(m_1) # 107.399764125016 (semble faire du sens...)
part2_finance_substitution(m_2) # 108.029950688959 (ne fait pas de sens...)
part2_finance_substitution(m_3) # 110.485123516122