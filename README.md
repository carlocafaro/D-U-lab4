# D-U-lab4


import numpy as np
import pylab as pl
from math import exp
def lmap(x):
   """
   Function to evaluate y = a*x + b for a = 2, b = 1

   inputs:
   x - a numpy array of values

   outputs:
   y - a numpy array containing the double well polynomial evaluated on x
   """
   a = 2
   b = 1
   return a*b + 1

def identity(x):
    return x
    

def exponential(x):

    return exp(x)

def square(x):

    return x**2
    
def myprior():
   """
   Function to draw a sample from a prior distribution given by the
   normal distribution with variance alpha.
   inputs: none
   outputs: the sample
   """
   alpha = 1.0
   return alpha**0.5*np.random.randn()

def Phi(f,x,y):
   """
   Function to return Phi, assuming a normal distribution for 
   the observation noise, with mean 0 and variance sigma.
   inputs:
   f - the function implementing the forward model
   x - a value of the model state x
   y - a value of the observed data
   """
   sigma = 1.0e-1
   return (f(x)-y)**2/(sigma**2)*2

def mcmc(f,prior_sampler,x0,yhat,N,beta):
   """
   Function to implement the MCMC pCN algorithm.
   inputs:
   f - the function implementing the forward model
   prior_sampler - a function that generates samples from the prior
   x0 - the initial condition for the Markov chain
   yhat - the observed data
   N - the number of samples in the chain
   beta - the pCN parameter

   outputs:
   xvals - the array of sample values
   avals - the array of acceptance probabilities
   """
   xvals = np.zeros(N+1)
   xvals[0] = x0
   avals = np.zeros(N)

   for i in range(N):
      
      prior_sampler=myprior()
      v=(1-beta**2)**(1/2)*xvals[i]+beta*prior_sampler
      avals[i]=min(1, np.exp(Phi(f, xvals[i], yhat)-Phi(f,v,yhat)))
      u=np.random.uniform()
      if u<avals[i]:
          xvals[i+1]=v
      else:
         xvals[i+1]=xvals[i]    

   return xvals,avals



def cumulative_average(avals,N):
    b=np.zeros(avals.shape)
    c=np.zeros(avals.shape)
    b=np.cumsum(avals)
    for i in range(N):
        c[i]=b[i]/(i+1)
       
    return c
    
    

  


if __name__ == '__main__':    

   "Exercise 1"
   N=1000
   beta=0.5
   xvals, avals= mcmc(identity,myprior,0.5,0.5,N,beta)
   c=cumulative_average(avals, N)
   
   "Here we discard the samples taken before the average acceptance rate has settled down" 
   "Let us fix the tolerance"
   tol=0.2
   for i in range(N):
        if np.abs(c[i]-0.25)>tol:
            xvals[i]=0.0
   pl.plot(range(N), c)
   pl.title('Cumulative average first exercise')  
   pl.show()
   mean=np.mean(xvals)
   variance=np.var(xvals)
   print ('The mean and variance of the samples after having discarded bad samples are respectively', mean, variance)
   
   
   pl.hist(xvals, bins=75, normed=1)
   pl.title('PDF of samples first exercise')
   pl.show()
   print('"Visualise the PDF of the samples. Does it look like a Normal distribution?')
   print('Yes, it looks like a Normal Distribution')
   
   
   
   "Exercise 2"
   N=1000
   
   
   
   "Let us tune beta and discard the samples taken before the average acceptance probability"
   yhat=1.0
   i=1
   tol=0.1
   c=np.zeros(len(range(N)))+1
   while(np.abs(c[N-1]-0.25)>tol):     
        beta=i*0.1
        xvals, avals= mcmc(exponential,myprior,0.5,yhat,N,beta)
        c=cumulative_average(avals, N) 
        i=i+1
   
   xvals, avals= mcmc(exponential,myprior,0.5,yhat,N,beta)
   for i in range(N):
        if np.abs(c[i]-0.25)>tol:
            xvals[i]=0.0
   
   xvals, avals= mcmc(exponential,myprior,0.5,yhat,N,beta)
   c=cumulative_average(avals, N)
   pl.plot(range(N), c)
   pl.title('Cumulative average 2nd exercise after having tuned beta and burned-in the algorithm')  
   pl.show()
   
   "Let us plot the PDF of the samples"
   pl.hist(xvals, bins=75, normed=1)
   pl.title('PDF of samples 2nd exercise')
   pl.show()
   
  
  
   "Exercise 3"
   N=1000
   
   "Let us tune beta and discard the samples taken before the average acceptance probability"
   yhat=1.0
   i=1
   tol=0.1
   c=np.zeros(len(range(N)))+1
   while(np.abs(c[N-1]-0.25)>tol):     
        beta=i*0.1
        xvals, avals= mcmc(square,myprior,0.5,yhat,N,beta)
        c=cumulative_average(avals, N) 
        i=i+1
   
   xvals, avals= mcmc(square,myprior,0.5,yhat,N,beta)
   for i in range(N):
        if np.abs(c[i]-0.25)>tol:
            xvals[i]=0.0
   
   "Here I am repeating four times mcmc calculations"
   
   for i in range(4):
      xvals, avals= mcmc(square,myprior,0.5,yhat,N,beta)
      c=cumulative_average(avals, N)
      pl.plot(range(N), c)
      pl.title('Cumulative average 3nd exercise after having tuned beta and burned-in the algorithm')  
      pl.show()
   
      "Let us plot the PDF of the samples"
      pl.hist(xvals, bins=75, normed=1)
      pl.title('PDF of samples 3nd exercise')
      pl.show()
   
   
