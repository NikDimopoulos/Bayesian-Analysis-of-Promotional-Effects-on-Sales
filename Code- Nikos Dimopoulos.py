import numpy as np
import matplotlib.pyplot as  plt
import pandas as pd
from scipy.stats import norm
from tabulate import tabulate

#Read the datasets
coupon=pd.read_excel("coupon.xls", engine="xlrd")
display=pd.read_excel("displ.xls", engine="xlrd")
price=pd.read_excel("price.xls", engine="xlrd")
sales=pd.read_excel("sales.xls", engine="xlrd")

#Take the dataset of the 71st brand
coupon = coupon.iloc[:, 70]
display = display.iloc[:, 70]
price = price.iloc[:, 70]
sales = sales.iloc[:, 70]

#take the logarithm of the sales and price
log_price = np.log(price)
log_sales = np.log(sales)

X=np.column_stack((np.ones((len(log_sales),1)),display,coupon,log_price))
y=log_sales


nos = 50000        # size of simulation sample
nob = 1000        # number of burn-in simulations

obs=len(y)
regressors=len(X[0])

print(obs,regressors)

#we create a matrix with the simulated parameters
draws = np.zeros((nos+nob,regressors+2))

#starting values of the gibbs sampler:
gamma=1
sigmasq=1
x = X.copy()

for i in range(nob+nos):

    if i%10000==0:
        print(i)


    #sample b|sigma^2, gamma, y
    bcov = np.linalg.inv(x.T @ x)  # (X'X)^{-1}
    cov=sigmasq*bcov #The covariance matrix sigma^2 * (X'X)^{-1}
    bhat = bcov @ x.T @ y  # OLS estimate of the coefficients
    beta = np.random.multivariate_normal(mean=bhat, cov=cov)
    #beta = (np.linalg.cholesky(cov)) @ np.random.normal(size=regressors) + bhat;# simulate beta from a normal distribution
    #with means bhat and covariance matrix cov

    #Simulate gamma| beta, sigma^2,
    #gamma follows a normal distribution
    #The first 62 observations do not include gamma, hence they do not affect the kernel of the distibution
    y2=y[62:]
    x2=x[62:,:]

    beta0s=np.full(62,beta[0])
    #create vector V that consists of 62 times beta 0 and -1
    V=np.concatenate([beta0s,[-1]])

    b0gamma_plus_residuals = y2 - x2[:,1:]@beta[1:]
    #Create vector W
    W=np.concatenate([b0gamma_plus_residuals,[-1]])
    gamma_mean = (1/(V.T @ V)) * (V.T @ W)
    gamma_var=sigmasq*(1/(V.T @ V))
    gamma=np.random.normal(gamma_mean, np.sqrt(gamma_var)) # Sample gamma from a normal distribution

    # we update the alpha_t on every iteration, since gamma changes
    alpha_t1 = np.ones(62)
    alpha_t2 = np.ones(62) * gamma


    x[:, 0] = np.concatenate([alpha_t1, alpha_t2])

    #sigma square
    res = y - x @ beta  # residuals y-Xβ
    #sigma square follows an interted gamma-2 distribution with parameter (y-Xβ)'(y-Xβ)+(gamma-1)^2
    parameter= res.T @ res + (gamma-1)**2
    #the degreees of freedom are N+1
    u = np.random.normal(size=len(y)+1)  # random draws of size N+1 from a normal distribution
    sigmasq =  parameter/ (u.T @ u )  # simulate sigma^2 by dividing the parameter by u'u  (u follows a normal distribution,hence υ'u~chisq(N)

    draws[i, regressors] = gamma
    draws[i, regressors+1]= sigmasq
    draws[i, 0:regressors] = beta  # store the simulated beta at the first columns of the matrix

# remove burn-in draws
draws = draws[nob:]

#thin value=1
draws = draws[range(0, nos , 1)]

print('Posterior Results');
print('Number of draws', (nos ) + nob);
print('Number of burn-indraws', nob);
print('Thin value', 1)


beta0mean=np.mean(draws[:,0])
beta1mean=np.mean(draws[:,1])
beta2mean=np.mean(draws[:,2])
beta3mean=np.mean(draws[:,3])
mean_gamma=np.mean(draws[:,4])
mean_sigmasq=np.mean(draws[:,5])

print("b0 mean: ", beta0mean)
print("b1 mean: ",beta1mean)
print("b2 mean: ",beta2mean)
print("b3 mean: ",beta3mean)
print("gamma mean: ",mean_gamma)
print("sigma squared mean: ",mean_sigmasq)

#define percentile generating function
def perc(y,p):
    draws=y+0
    if (p<1) and (p>0):
        draws=np.sort(draws,axis=0)
    else:
        draws[:]=np.nan
    return(draws[round(p*len(draws))-1])

# create parameter names for our table
varnames=[]
for i in range(regressors):
   varnames.append('beta ' + str(i))
varnames.append('gamma')
varnames.append('sigma^2')

#Calculate the means of the parameters
postmean=(np.mean(draws, axis=0))

#Create the table
table=np.column_stack((varnames, perc(draws,0.10), postmean, perc(draws,0.90)))
titles=["parameter","10% percentile","mean","90% percentile"]
table = tabulate(table, titles, tablefmt='pretty')
print(table)

gamma_draws = draws[:, regressors] #the gammas lie on the 5-th column
post_pr_gamma_smaller_1 = np.mean(gamma_draws < 1)
print("Probability(gamma<1) = ", post_pr_gamma_smaller_1 )
