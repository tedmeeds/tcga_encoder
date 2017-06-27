import time
from tcga_encoder.analyses.biclustering.models import *
from tcga_encoder.analyses.biclustering.parameters import *
from tcga_encoder.analyses.biclustering.bic_data import *
from tcga_encoder.analyses.biclustering.bic_math import *

class Biclustering(object):
  def __init__(self, iters = 20, type = "gauss", maximize = True ):
    self.simulationspecs          = SimulationSpecs()
    self.type = type
        
    self.simulationspecs.nMCMC    = iters
    self.simulationspecs.nBurnIn  = 0
    self.simulationspecs.nSplitMergeMove = 5
    self.simulationspecs.nAdjustedMove   = 0
    self.simulationspecs.nSamplesForNew  = 5
    self.simulationspecs.bMaximize       = maximize

    self.clusteringPrior                  = ClusteringPrior()
    self.clusteringPrior.bUseRowBias      = 1   # use row cluster parameters
    self.clusteringPrior.bUseColumnBias   = 1   # use column cluster parameters
    self.clusteringPrior.bUseBicluster    = 1   # use block-constant bicluster parameters
    self.clusteringPrior.row_hyper        = ClusterHyperPrior( conc_a = 1.5, conc_b = 1.0 )
    self.clusteringPrior.col_hyper        = ClusterHyperPrior( conc_a = 1.5, conc_b = 1.0 )
    self.clusteringPrior.rowPrior         = ClusterPrior( conc = 15.0, discount = 0.0 ) 
    # concentration/discount parameter for the row Pitman-Yor process
    self.clusteringPrior.colPrior         = ClusterPrior( conc = 15.0, discount = 0.0 )


    self.modelPrior                   = BiclusteringPrior()
    #modelPrior.biclusterType     = 'ConjugateGaussianBicluster'
    self.modelPrior.biclusterType     = 'NonConjugateGaussianBicluster'
    self.bSparse = False
    self.modelPrior.offset_hyper      = GaussianHyperparams( dof_a = 10.0, dof_b = 1.0, expprec_a = 10.0, expprec_b = 1.0 )
    self.modelPrior.bic_hyper         = GaussianHyperparams( dof_a = 1.0, dof_b = 1.0, expprec_a = 1.0, expprec_b = 1.0 )
    self.modelPrior.row_hyper         = GaussianHyperparams( dof_a = 1.0, dof_b = 1.0, expprec_a = 1.0, expprec_b = 1.0 )
    self.modelPrior.col_hyper         = GaussianHyperparams( dof_a = 1.0, dof_b = 1.0, expprec_a = 1.0, expprec_b = 1.0 )
    self.modelPrior.rowoffset_params  = GaussianParams( mu_base = 0.0, degreesOfFreedom = 2.0, expPrec = 1.0, dataPrecConst = 1.0, muPrecConst = 1.0, bSparse = self.bSparse )
    self.modelPrior.coloffset_params  = GaussianParams( mu_base = 0.0, degreesOfFreedom = 2.0, expPrec = 1.0, dataPrecConst = 1.0, muPrecConst = 1.0, bSparse = self.bSparse )
    self.modelPrior.bicluster_params  = GaussianParams( mu_base = 0.0, degreesOfFreedom = 2.0, expPrec = 1.0, dataPrecConst = 1.0, muPrecConst = 1.0, bSparse = self.bSparse )
    self.modelPrior.rowcluster_params = GaussianParams( mu_base = 0.0, degreesOfFreedom = 2.0, expPrec = 1.0, dataPrecConst = 1.0, muPrecConst = 1.0, bSparse = self.bSparse )
    self.modelPrior.colcluster_params = GaussianParams( mu_base = 0.0, degreesOfFreedom = 2.0, expPrec = 1.0, dataPrecConst = 1.0, muPrecConst = 1.0, bSparse = self.bSparse )
    
    
  def fit( self, X ):
    self.dataset = BiclusteringFullData( X )
    self.M = BuildModel( self.modelPrior, self.clusteringPrior, self.dataset  ) 
    self.M.Init()
    self.M.simspecs = self.simulationspecs
    logprobs = []
    for t in range(self.simulationspecs.nMCMC):
        t0 = time.clock() 
        self.M.RunGibbs(t)
        self.M.UpdateHyperParameters()
        t1 = time.clock()
        print t, "TIME: ",t1-t0 , self.M.LogProbData
        logprobs.append( self.M.LogProbData)

        self.row_labels_ = self.M.U
        self.col_labels_ = self.M.V
        #self.M.Show(self.simulationspecs.outdir, t)
        
        # U = M.U
        # V = M.V
        #
        # write_class_ids( U, outfile_u )
        # write_class_ids( V, outfile_v )
        #pp.show()
    return self
    
    