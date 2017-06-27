
#from Math.Function      import Gammaln, LogFactorial, LogSum as logsumexp
from numpy import *
from numpy.random import rand, exponential
LOG2PI  = log( 2.0 * pi )

#from PythonPackages.Math.Function import FastRndWeave
from numpy import zeros, array, random, matrix
from numpy.random import rand
GLN_CONST1 = 0.5*log( 2*pi )
GLN_CONST2 = 0.5*log(pi)


def safecorrcoef( X, O = None ):
    # X: data matrix
    # O: matrix of observations
    CC = 999.0*ones( (X.shape[0], X.shape[0]), dtype=float)
    if O is not None:
        for i,xi,oi in zip( range( X.shape[0] ), X, O ):
            for j,xj,oj in zip( range( X.shape[0] ), X, O ):
                o = oi*oj
                CC[i][j] = correlate(xi[o], xj[o])[0] #dot( xi[o], xj[o] ) / sqrt(dot( xi[o], xi[o] )*dot( xj[o], xj[o] ))
    else:
        for i,xi in zip( range( X.shape[0] ), X):
            oi = xi != 999.0
            for j,xj in zip( range( X.shape[0] ), X ):
                oj = xj != 999.0
                o = oi*oj
                CC[i][j] = correlate(xi[o], xj[o])[0] #dot( xi[o], xj[o] ) / sqrt(dot( xi[o], xi[o] )*dot( xj[o], xj[o] ))
    return CC
def sortrows(x,col=None):
  #SORTROWS Sort rows in ascending order.
  #   SORTROWS(X) sorts the rows of the matrix X in ascending order as a
  #   group.  For ASCII strings, this is the familiar dictionary sort.
  #   When X is complex, the elements are sorted by ABS(X). Complex
  #   matches are further sorted by ANGLE(X).
  #
  #   SORTROWS(X,COL) sorts the matrix based on the columns specified in
  #   the vector COL.  For example, SORTROWS(X,[2 3]) sorts the rows of X
  #   by the second and third columns of X.
  #
  #   [Y,I] = SORTROWS(X) also returns an index matrix I such that Y = X(I,:).
  #
  #   See also SORT.

  #   Copyright 1984-2000 The MathWorks, Inc.
  #   $Revision: 1.13 $  $Date: 2000/06/01 02:10:40 $

  (m,n) = shape(x)
  if col is None:
      col = arange(n-1,-1,-1)

  # Sort back to front
  if m>0:
    ndx = arange(0,m,1).T
  else:
    ndx = []

  #pdb.set_trace()
  for ci in col: #=length(col):-1:1,

    ind = argsort(x[ndx,ci])
    #print x[ndx,ci], ind, ndx    
    #pdb.set_trace()
    ndx = ndx[ind]

  y = x[ndx]
  return y

def erf( x ):
   x2 = x*x
   z = array( [1.0, -1.0/3.0, 1.0/10.0, -1.0/42.0, 1./216.] )
   y = zeros( 5 )
   y[0] = x
   y[1] = x*x2
   y[2] = y[1]*x2
   y[3] = y[2]*x2
   y[4] = y[3]*x2
   w = z*y
   return 2.0*w.sum()/SQRT2

def GlobalLog( x ):
    global logapprox
    try:
        return logapprox[x]
    except IndexError:
        ln = len(logapprox)
        logapprox = hstack( (logapprox, log( arange(x-ln+1)+ln ) ) )
        return logapprox[x.astype(int)]
        
    except TypeError:
        try:
            return logapprox[x.astype(int)]
        except NameError:
            print 'new log approx'
            logapprox = log( arange(x.astype(int)+1) )
            return logapprox[x.astype(int)]
        else:
            assert False, 'dfgdfg'
    except NameError:
        print 'new log approx'
        logapprox = log( arange(x.astype(int)+1) )
        return logapprox[x.astype(int)]
    else:
        assert False, 'dfgdfg'

def GlobalLog2( x ):
    global logapprox
    try:
        return logapprox[x]
    except TypeError:
        try:
            return logapprox[x.astype(int)]
        except NameError:
            print 'new log approx'
            mx = ceil(max(x.astype(int)))
            logapprox = log( arange(mx+1) )
            return logapprox[x.astype(int)]
        except IndexError:
            #if x.dtype != int:
            #pdb.set_trace()
            mx = ceil(max(x.astype(int)))
            ln = len(logapprox)
            logapprox = hstack( (logapprox, log( arange(mx-ln+1)+ln ) ) )
            return logapprox[x.astype(int)]
        else:
            assert False, 'Unknown exception'
    except IndexError:
            #if x.dtype != int:
            #pdb.set_trace()
            mx = ceil(max(x.astype(int)))
            ln = len(logapprox)
            logapprox = hstack( (logapprox, log( arange(mx-ln+1)+ln ) ) )
            return logapprox[x.astype(int)]
    except NameError:
        print 'new log approx'
        try: 
            mx = ceil(max(x))
        except:
            mx = x
        logapprox = log( arange(mx+1) )
        try:
            return logapprox[x]
        except IndexError:
            #if x.dtype != int:
            #pdb.set_trace()
            mx = ceil(max(x.astype(int)))
            ln = len(logapprox)
            logapprox = hstack( (logapprox, log( arange(mx-ln+1)+ln ) ) )
            return logapprox[x.astype(int)]
        else:
            return logapprox[x.astype(int)]
    else:
        assert False, 'Unknown exception'

"""
    An approximation to the natural log of Gamma(x).  
    Uses Stirling approximation.
"""
def LogGammaStirling(x):
    y = (x-0.5)*log(x) - x + GLN_CONST1
    return y

def LogFactorialStirling(x):
    y = (x+0.5)*log(x) - x + GLN_CONST1
    return y
    
"""
    An approximation to the natural log of Gamma(x).  
    Uses Ramanujan approximation.
"""
def LogFactorialRamanujan(x):
    y = x*log(x) + (1.0/6.0)*log(x*(1.+4.*x*(1.+2.*x)))- x + GLN_CONST2
    #print log(x)
    return y
    
def Gammaln(x):
    return LogGammaStirling(x)
    #return LogFactorialRamanujan(x-1)
    
def LogPartition( conc, cnts ):
    N = cnts.sum()
    K = len(cnts)
    logprob = K * log( conc ) - Gammaln( conc + N ) + Gammaln( cnts ).sum()
    return logprob

def LogFactorial(x):
    try:
        n = x.size
    except:
        n = 1
    if n == 1:
        if x > 0:
            return LogFactorialRamanujan(x)
        else:
            return 0.0
    
    y = zeros( x.size, dtype=float )
    iok = x!=0
    y[iok] = LogFactorialRamanujan( x[iok] )
    return y

def LogRnd( x ):
    lr = exp(x)
    # ind = FastRndWeave( rand(), lr )
    ind = FastRnd( rand(), lr )
    #pdb.set_trace()
    return ind

# def FastRndWeave( randval,r ):
#     from scipy import weave
#     #v = array([r[0]])
#     newk = array([0],dtype=int)
#     code = r"""
#     //int newk = 0;
#     double v = r(0);
#     while (1)
#     {
#         if (v >  (double) randval)
#             break;
#         else{
#             newk(0) = newk(0)+1; //++;
#             v += r(newk(0));}
#         //printf( "%d %.2f\n", newk(0), v );
#     }
#     //return_val = newk;
#     """
#     vars = ['randval','r','newk']
#     global_dict={'randval':randval, 'r':r,'newk':newk}
#     weave.inline(code,vars,global_dict=global_dict,\
#                    type_converters=weave.converters.blitz)
#     return newk[0]

def FastLogRnd( x ):
    return FastRnd( rand(), exp( x ) )

def FastRnd( randval,r ):
    newk = 0 # the index in discrete distribution sampled
    v = r[0]
    while True:
        if v >  randval:
            break
        else:
            newk += 1
            v += r[newk]
    return newk

def LogSum(x,xmax=None,dim=0):
  """Compute log(sum(exp(x))) in numerically stable way."""
  #xmax = x.max()
  #return xmax + log(exp(x-xmax).sum())
  if dim==0:
    #print xmax
    if xmax is None: xmax = x.max(0)
    return xmax + log(exp(x-xmax).sum(0))
  elif dim==1:
    if xmax is None: xmax = x.max(1)
    return xmax + log(exp(x-xmax[:,newaxis]).sum(1))
  else: raise 'dim ' + str(dim) + 'not supported'

def LogSumWeave(x,xmax=None):
  """Compute log(sum(exp(x))) in numerically stable way."""
  if xmax is None:
      xmax = max(x)
  from scipy import weave
  code = """
        #include <math.h>
        //x=x-xmax;
        //x = exp(x);
        //double ls = sum(x);
        //ls = log(ls) + xmax;        
        double ls = xmax + log(sum( exp(x-xmax) ) );
        return_val = ls;
  """
  vars = ['x','xmax']
  global_dict={'x':x, 'xmax':xmax}
  return weave.inline(code,vars,global_dict=global_dict,\
                   type_converters=weave.converters.blitz)

def LogProbRnd( lp ):
   return Discrete( exp( lp - LogSum(lp) ) ).Rnd()

def LogProbNormedRnd( lp ):
   return Discrete( exp( lp ) ).Rnd()

""" a la West """
def ConcentrationRnd( a, b, oldconc, N, K, reps = 1 ):
    from Distribution.Dirichlet import Dirichlet
    for t in range(reps):
        # sample x ~ Beta( oldconc + 1, N )
        # only need log(x)
        assert N > 0, 'N is zero or neg: ' + str(N)
        assert oldconc+1 > 0, 'odconc is zero or neg: ' + str(oldconc)
        logx = Dirichlet( array([oldconc + 1, N]) ).Rnd().LogProb(0)
        # y ~ Pi_1,Pi_2
        # Pi_1 \propto a+K-1, Pi_2 \propto N*(b-logx)
        assert a+K-1 > 0, 'a+K-1 is zero or neg: ' + str(a+K-1)
        assert oldconc+1 > 0, 'odconc is zero or neg: ' + str(oldconc)
        y = Discrete( array([a+K-1, N*(b-logx)]) ).Rnd()

        if y == 0:
            oldconc = gamma( a+K, b-logx )
        else:
            oldconc = gamma( a+K-1, b-logx )
    return oldconc

def TablesInCRP( p ):
    c = zeros( len(p), dtype=int ) 
    k = 0
    for (w,n) in p:
        if n > 0:
            c[k] = 1
            if n > 1:
                R = random.rand(n-2)
                W = float(w) / (w+arange(2,n)-1)
                B = R<W
                B.astype(int)
                c[k] += B.sum()
        k+=1
    return c
def TablesInCRP2( p ):
    c = zeros( len(p), dtype=int ) 
    k = 0
    N = 0
    for (w,n) in p:
        N += n
    R = random.rand(N)
    r = 0
    for (w,n) in p:
        #print w
        #print n
        if n > 0:
            #print arange(2,n)
            c[k] = 1
            for i in arange(2,n):
                #print  float(w)/(w+i-1)
                if R[r] < float(w)/(w+i-1):  
                    c[k] +=1
                    r += 1
        k+=1
    return c

def PitmanYor( N, conc, discount ):
    assert conc > 0.0, "concentration bad: " + str(conc)
    assert discount >= 0.0, "discount bad: " + str(discount)
    assert discount < 1.0, "discount bad: " + str(discount)

    Z    = zeros( N, dtype = int )
    cnts = []
    K    = 0
    R = rand(N)
    logprob = 0.0
    for i in range( N ):
        P       = zeros( K + 1, dtype = float ) 
        P[-1]   = conc + K*discount
        P[0:-1] = array( cnts, dtype = float ) - discount
        P       = P / sum(P)
        #Z[i]    = FastRndWeave( R[i], P )
        Z[i]    = FastRnd( R[i], P )

        if Z[i] == K:
            K += 1
            cnts.append( 1 )
        else:
            cnts[Z[i]] += 1
    return Z,cnts

""" Generating functions """
def PitmanYorOld( N, conc, discount ):
    U = zeros( N, dtype = int )
    U[0] = 1
    P = zeros( 2, dtype = float )
    P[0] = 1.0
    P[1] = conc
    K = 1
    for i in range(N-1):
        pk = P / P.sum()
        k = rand(pk)
        U[i+1] = k
        if k == K:  
            # a new cluster
            P[k] = 1.0
            P.extend(1)
            P[-1] = conc
            K += 1
        else:
            P[k] += 1.0
    return U
            

""" PROB DISTRIBUTIONS LIKELIHOODS """
def GaussianLogProb( x, m, v ):
   dim     = len(x)
   logprob = -0.5 * dim * LOG2PI - 0.5 * log( v )
   logprob += -0.5*v*((x-v)**2).sum()
   return logprob

#def StudentLogProb( x, m, N, a, b, c, dof ):
    
def DiscreteLogProb( x, w ):
    logw = log(w)
    return logw[x].sum()

def GammalnLogProb( x, n, phi ):
    logprob = 0.0
    nx = len( x )
    nn = n.sum()
    V  = len( n )
    logprob = Gammaln( phi + nn ) - Gammaln( phi + nn + nx )
    for nv,v in zip( n, range( len(n) ) ):
        nvx = len( find( x == v ) )
        logprob += Gammaln( phi + nv + nvx ) - Gammaln( phi + nv )
    return logprob

def slice_sample(logdist,params,xinit,L,R,W,N,MODE):
    """
    % Description
    % -------------------
    % Use slice sampling to generate samples from any distribution over
    % a one dimensional space where the initial interval [l,r] is known.
    % Takes a one parameter function computing the log of the distribution,
    % or any function computing the log of a function proportional to the
    % distribution. Additional parameters and constants can be passed in
    % using the params structure.
    %
    % Syntax: x=slice_sample(logdist,params,xinit,l,r,n)
    % --------------------
    % * logdist: a function handle for a function computing the log ofa distribution of
    %           one parameter.
    % * params:  a structure of paramaeters and constants needed by logdist
    % * xinit:   initial point to start sampling from
    % * L:       the lower bound of the sampling interval
    % * R:       the upper bound of the sampling interval
    % * N:       the number of samples to draw
    %* MODE:    0 - perform shrinkage on the given interval
    %           1 - perform stepping out then shrinkage
    %*          2 - perform doubling then shrinkage
    """
    eps = 0.00001
    #declare space for samples
    x = zeros(N, dtype = float)
    x[0] = xinit

    #maximum times to expand or shrink the interval
    maxsearch=10

    #sample n points from the distribution
    #pdb.set_trace()
    for i in arange( 1, N, 1 ):
        #pick the slice level from a uniform density under the logposterior curve
        logprob_old = logdist(x[i-1],params)
        assert isnan(logprob_old) == False, 'Slice Error: logdist returned NaN'
        z = logprob_old-exponential(1.0)

        #Determine the interval
        if MODE==0:  # shrinkage on the given interval
            l=L
            r=R
        elif MODE==1: # stepping out
            c = rand()
            l=x[i-1]-c*W
            if l <= L:
                l = L+eps
            r=l+W
            if r >= R:
                r = R-eps

            logprobl = logdist( l, params)
            assert isnan(logprobl) == False, 'Slice Error: logdist returned NaN'
            j=0
            while logprobl > z and j<maxsearch:
                l-=W
                if l <= L:
                    l = L+eps
                    break
                logprobl = logdist( l, params)
                assert isnan(logprobl) == False, 'Slice Error: logdistreturned NaN'
                j=j+1

            logprobr = logdist( r, params)
            assert isnan(logprobr) == False, 'Slice Error: logdist returned NaN'
            j=0;
            while logprobr > z and j<maxsearch:
                r+=W
                if r >= R:
                    r = R-eps
                    break
                logprobr = logdist( r, params)
                assert isnan(logprobr) == False, 'Slice Error: logdistreturned NaN'
                j=j+1

        elif MODE==2: # doubling
            c = rand()
            l=x[i-1]-c*W
            if l < L+eps:
                l = L+eps
            r=l+W
            if r > R:
                r = R-eps

            logprobl = logdist( l, params)
            assert isnan(logprobl) == False, 'Slice Error: logdist returned NaN'
            logprobr = logdist( r, params)
            assert isnan(logprobr) == False, 'Slice Error: logdist returned NaN'
            j=0
            while (j<maxsearch and (logprobl > z or logprobr > z) and (abs(r-R)>2*eps or abs(l-L)>2*eps )):
                c=rand()
                j=j+1
                if (c<0.5 or R==r) and l>L:
                    l -= (r-l)
                    if l<=L:
                        l=L+eps
                    logprobl = logdist(l, params)
                    assert isnan(logprobr) == False, 'Slice Error:logdist returned NaN'
                elif (c>=0.5 or L==l) and r<R:
                    r +=  (r-l)
                    if r>=R:
                        r=R-eps
                    logprobr = logdist(r, params)
                    assert isnan(logprobr) == False, 'Slice Error:logdist returned NaN'

        #shrink until we draw a good sample
        j=0
        while j < maxsearch:
            j=j+1
            # randomly sample a new alpha on the interval
            x_new = l + rand()*(r-l)

            # compute the log posterior probability for the new alpha
            # any function proportional to the true posterior is fine
            logprob  = logdist( x_new,params)
            assert isnan(logprob) == False, 'Slice Error: logdist returned NaN'

            # Accept the sample if the log probability lies above z
            if logprob>z or abs(r-l)<eps:
                x[i] = x_new
                break
            else:
                # If not, shrink the interval
                if x_new<x[i-1]:
                    l = x_new
                else:
                    r = x_new
        # check to see if a new value was assigned.
        # if not assign the previous value and we try again.
        if x[i]==0 and j==maxsearch:
            x[i] = x[i-1]
    return x, logprob