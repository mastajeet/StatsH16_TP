import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as stt
import statsmodels.api as sti
import helper
import scipy.stats.distributions as dist
from collections import defaultdict
import itertools as it
import numpy as np

# Actuariat

data = pd.read_csv('./data/dobson.csv')
data['claim_rate'] = data['n']/data['y']

# a)

plt.scatter(data['age'],data['claim_rate'],c='b',marker='o')
plt.title('Age Category vs Claim Rate')
plt.scatter(data['car'],data['claim_rate'],c='r',marker='+')
plt.title('Car Category vs Claim Rate')
plt.scatter(data['dist'],data['claim_rate'],c='g',marker='x')
plt.title('District vs Claim Rate')

# b)
all_results = defaultdict(dict)

all_regressions = [
    "claim_rate ~ C(age) + C(car) + C(dist) + C(age):C(car) + C(age):C(dist) + C(car):C(dist) ",
    "claim_rate ~ C(age) + C(car) + C(age):C(car)",
    "claim_rate ~ C(age) + C(car) + C(age):C(car)",
    "claim_rate ~ C(age) + C(dist) + C(age):C(dist)",
    "claim_rate ~ C(car) + C(dist) + C(car):C(dist)",
    "claim_rate ~ C(car)",
    "claim_rate ~ C(dist)",
    "claim_rate ~ C(age)",
]

for reg in all_regressions:
    regression = stt.poisson(reg,data=data)
    results = regression.fit()
    helper.wrap_regression_results(all_results,results)

results = pd.DataFrame(all_results).transpose()
combinaisons = it.permutations(range(7),2)

for pair in combinaisons:
    a,b = pair
    if results.iloc[a].degree_of_freedom > results.iloc[b].degree_of_freedom:
        Deviance = -2*(results.iloc[a].log - results.iloc[b].log)
        Degree_of_freedom = results.iloc[a].degree_of_freedom - results.iloc[b].degree_of_freedom
        if Deviance < dist.chi2.ppf(0.95,Degree_of_freedom):
            print(results.iloc[a].name, 'est rejete pour', results.iloc[b].name)

# Best regression : claim_rate ~ C(dist)

# c)

best_regression_b = stt.poisson("claim_rate ~ C(dist) ",data=data)
best_results_b = best_regression_b.fit()

test_regression_c = stt.poisson("claim_rate ~ age + car",data=data)
test_results = test_regression_c.fit()

deviance = -2 * (test_results.llf - best_results_b.llf )
degree_of_freedom = test_results.df_model - best_results_b.df_model

if deviance < dist.chi2.ppf(0.95,degree_of_freedom):
    print('test_regression_c est rejete pour best_regression_b')



# Finance
# a)
r_royal = helper.yf_log_yield_extractor('./data/royalbank_monthly.csv',number_datapoint=30)
r_royal.to_excel('./output/log_yield_royalbank.xlsx')

# b)
plt.scatter([x for x in range(30)],r_royal['Yield']**2)
plt.title('Yield squared vs Time')

# c)
plt.scatter([x for x in range(30)],r_royal['Yield'])
plt.title('Yield vs Time')


mu = np.mean(r_royal['Yield'])

y = (r_royal['Yield'][:29]-mu)**2
x = (r_royal['Yield'][1:30]-mu)**2
x = sti.add_constant(x)

regression = stt.OLS(list(y),np.array(x))
results = regression.fit()
results.summary()

h31 = results.predict((1,y[0]))[0]
h32 = results.predict((1,h31))[0]









