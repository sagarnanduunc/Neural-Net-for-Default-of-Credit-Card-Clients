""" Gradient Descent optimization tools 
    
    Steepest descent and scaled conjugate gradient 
    are currently implemented. 
    Reference codes are R version of Chuck Anderson's.
    
    Reference
        Chong and Zak (2008).
          http://www.engr.colostate.edu/~echong/book3/
        Moller (1993).
          http://www.sciencedirect.com/science/article/pii/S0893608005800565

                                by lemin (Minwoo Jake Lee)
        
    last modified: 10/01/2011
"""

import numpy as np
import math
from copy import copy 
import pdb
from math import sqrt, ceil
import sys
import inspect


def lineno():
    """Returns the current line number in our program."""
    #print "::: ", inspect.currentframe().f_back.f_lineno
    return inspect.currentframe().f_back.f_lineno

###########################################################
###
### Scaled Conjugate Gradient 
###
###########################################################

def scg(w, gradf, optimf, *fargs, **params):

    wprecision = params.pop("wPrecision",1.e-8)
    fprecision = params.pop("fPrecision",1.e-8)
    niter = params.pop("nIterations",1000)

    wtracep = params.pop("wtracep",False)
    ftracep = params.pop("ftracep",False)

    bwmin = params.pop("bwmin",False)
    wmin  = params.pop("wmin",1)
    bwmax = params.pop("bwmax",False)
    wmax  = params.pop("wmax",1)

    bverbose = params.pop("verbose", False)
    _beta = params.pop("beta", 1e-6)

    while True: # outmost loop to restart for RGD
        nvars = len(w)
        sigma0 = 1.0e-10
        fold = optimf(w, *fargs)
        fnow = fold
        #f = fnow
        gradnew = gradf(w, *fargs)
        gradold = copy(gradnew)
        d = -gradnew				# Initial search direction.
        success = True				# Force calculation of directional derivs.
        nsuccess = .0				# nsuccess counts number of successes.
        beta = _beta				# Initial scale parameter.
        betamin = 1.0e-15 			# Lower bound on scale.
        betamax = 1.0e5			# Upper bound on scale.
        #betamax = 1.0e20			# Upper bound on scale.
        j = 1				# j counts number of iterations.

        wtrace = []
        if wtracep:
            wtrace.append(w)
        ftrace = []
        if ftracep:
            ftrace.append(fnow) #[0,0])
        
        while j <= niter:

            ## Calculate first and second directional derivatives.
            if success:
                mu = np.dot(d,gradnew)
                if bverbose and np.isnan(mu): print("mu is NaN")
                if mu >= 0:
                    d = - gradnew
                    mu = np.dot(d,gradnew)

                kappa = np.dot(d,d)
                if kappa < np.finfo(np.double).eps:
                    return {'w':w, 
                            'f':fnow, 
                            'reason':"limit on machine precision",
                            'wtrace':wtrace if wtracep else None, 
                            'ftrace':ftrace if ftracep else None }

                sigma = sigma0/math.sqrt(kappa)
                wplus = w + sigma * d
                gplus = gradf(wplus, *fargs)
                theta = np.dot(d, (gplus - gradnew))/sigma

            ## Increase effective curvature and evaluate step size alpha.
            delta = theta + beta * kappa
            if delta is np.nan: print("delta is NaN")
            if delta <= 0.:
                delta = beta * kappa
                beta = beta - theta/kappa
            
            alpha = -mu/delta
            #print "alpha:", alpha
            ## Calculate the comparison ratio.
            wnew = w + alpha * d
            fnew = optimf(wnew, *fargs)

            if bwmin and all(wnew <= wmin): 
                return {'w':w, 
                        'f':fnow, 
                        'reason':"limit on w min.(%f)" % wmin,
                        'wtrace':wtrace if wtracep else None, 
                        'ftrace':ftrace if ftracep else None }
            if bwmax and all(wnew >= wmax):
                return {'w':w, 
                        'f':fnow, 
                        'reason':"limit on w max.(%f)" % wmax,
                        'wtrace':wtrace if wtracep else None, 
                        'ftrace':ftrace if ftracep else None }

            Delta = 2. * (fnew - fold) / (alpha*mu)
            if not np.isnan(Delta) and Delta >= 0.:
                success = True
                nsuccess = nsuccess + 1
                w = wnew
                fnow = fnew
                if wtracep:
                    wtrace.append(w)
                if ftracep:
                    ftrace.append(fnow) #[0,0])
            else:
                success = False
                fnow = fold
            #f = fnow

            #if (j % (niter /10)) ==0 and bverbose:
            #    print "SCG: Iteration %d  f=%f  scale=%f" % (j, fnow, beta)

            if success:
                ## Test for termination
                if np.max(abs(alpha*d)) < wprecision:
                    return {'w':w, 
                            'f':fnow, 
                            'reason':"limit on w Precision",
                            'wtrace':wtrace if wtracep else None, 
                            'ftrace':ftrace if ftracep else None }
                elif np.max(abs(fnew-fold)) < fprecision:
                    return {'w':w, 
                            'f':fnow, 
                            'reason':"limit on f Precision",
                            'wtrace':wtrace if wtracep else None, 
                            'ftrace':ftrace if ftracep else None }
                else:
                    ## Update variables for new position
                    fold = fnew
                    gradold = gradnew
                    gradnew = gradf(w, *fargs)
                    ## If the gradient is zero then we are done.
                    if np.dot(gradnew, gradnew) == 0:
                        return {'w':w, 
                                'f':fnow, 
                                'reason':"zero gradient",
                                'wtrace':wtrace if wtracep else None, 
                                'ftrace':ftrace if ftracep else None }
              
            ## Adjust beta according to comparison ratio.
            if np.isnan(Delta) or Delta < 0.25:
                beta = min(4.0*beta, betamax)
            elif Delta > 0.75:
                beta = max(0.5*beta, betamin)

            ## Update search direction using Polak-Ribiere formula, or re-start 
            ## in direction of negative gradient after nparams steps.

            if nsuccess == nvars:
                d = -gradnew
                nsuccess = 0
            elif success:
                gamma = np.dot(gradold - gradnew, gradnew / mu)
                d = gamma * d - gradnew

            j = j + 1

        return {'w':w, 
                'f':fnow, 
                'reason':"reached limit of nIterations",
                'wtrace':wtrace if wtracep else None, 
                'ftrace':ftrace if ftracep else None }



######################################################################
### Steepest descent
    

floatPrecision = sys.float_info.epsilon

def steepest(x,gradf, f, *fargs, **params):
    """steepest:
    Example:
    def parabola(x,xmin,s):
        d = x - xmin
        return np.dot( np.dot(d.T, s), d)
    def parabolaGrad(x,xmin,s):
        d = x - xmin
        return 2 * np.dot(s, d)
    center = np.array([5,5])
    S = np.array([[5,4],[4,5]])
    firstx = np.array([-1.0,2.0])
    r = steepest(firstx, parabola, parabolaGrad, center, S,
                 stepsize=0.01,xPrecision=0.001, nIterations=1000)
    print('Optimal: point',r[0],'f',r[1])"""

    stepsize= params.pop("stepsize",0.1)
    evalFunc = params.pop("evalFunc",lambda x: "Eval "+str(x))
    nIterations = params.pop("nIterations",1000)
    xPrecision = params.pop("xPrecision",1.e-8)
    fPrecision = params.pop("fPrecision",1.e-8)
    wtracep = params.pop("wtracep",False)
    ftracep = params.pop("ftracep",False)

    i = 1
    if wtracep:
        wtrace = np.zeros((nIterations+1,len(x)))
        wtrace[0,:] = x
    else:
        wtrace = None
    oldf = f(x,*fargs)
    if ftracep:
        ftrace = [0] * (nIterations+1) #np.zeros(nIterations+1)
        ftrace[0] = f(x,*fargs)# [0,0]
    else:
        ftrace = None
  
    while i <= nIterations:
        g = gradf(x,*fargs)
        newx = x - stepsize * g
        newf = f(newx,*fargs)
        #if i % max(1,(nIterations/10)) == 0:
        #    print "Steepest: Iteration",i,"Error",evalFunc(newf)
        if wtracep:
            wtrace[i,:] = newx
        if ftracep:
            ftrace[i] = newf #[0,0]

        if np.any(newx == np.nan) or newf == np.nan:
            raise ValueError("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
        if np.any(np.isinf(newx)) or  np.isinf(newf):
            raise ValueError("Error: Steepest descent produced newx that is NaN. Stepsize may be too large.")
        if max(abs(newx - x)) < xPrecision:
            return {'w':newx, 'f':newf, 'nIterations':i, 'wtrace':wtrace[:i,:] if wtracep else None, 'ftrace':ftrace[:i] if ftracep else None,
                    'reason':"limit on x precision"}
        if abs(newf - oldf) < fPrecision:
            return {'w':newx, 'f':newf, 'nIterations':i, 'wtrace':wtrace[:i,:] if wtracep else None, 'ftrace':ftrace[:i] if ftracep else None,
                    'reason':"limit on f precision"}
        x = newx
        oldf = newf
        i += 1

    return {'w':newx, 'f':newf, 'nIterations':i, 'wtrace':wtrace[:i,:] if wtracep else None, 'ftrace':ftrace[:i] if ftracep else None, 'reason':"did not converge"}


