""" 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    GAUSSIAN PARAMS CLASS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
class GaussianParams( object ):
    def __init__(self, mu_base = 0.0, degreesOfFreedom = 1.0, expPrec = 1.0, dataPrecConst = 1.0, muPrecConst = 1.0, bSparse = False  ):
        self.mu = mu_base
        self.nu = degreesOfFreedom
        assert self.nu > 0.0, 'gone bad'
        self.a  = dataPrecConst
        self.b  = muPrecConst
        self.c  = expPrec
        self.bSparse = bSparse
        print self.nu,self.a,self.b,self.c

class GaussianHyperparams( object ):
    def __init__(self, dof_a = 5.0, dof_b = 1.0, expprec_a = 5.0, expprec_b = 1.0  ):
        self.dof_a      = dof_a
        self.dof_b      = dof_b
        self.expprec_a  = expprec_a
        self.expprec_b  = expprec_b