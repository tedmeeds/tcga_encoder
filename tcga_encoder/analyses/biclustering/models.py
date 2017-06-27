from bic_math import PitmanYor, LOG2PI, LogSum, LogRnd, Gammaln, slice_sample, LogPartition, LogSumWeave
from matplotlib.mlab import find
#from Graphviz.Dotty import *
# from scipy import weave

LOGMIN  = -743.746924741
from numpy import *
from pylab import hist, matshow, cm, hlines, legend,vlines, show, draw, axis, figure, ion, ioff, clf, hold, colorbar, prism,autumn, plot,imshow
import pdb
from scipy import special
MISSINGVAL = 999.0

DLEN = 3

GAMMALN = {}
SMALLGAMMALN = {}
LOG = {}
SMALLLOG = {}
def GETGAMMALN( x ):
    return special.gammaln( x +0.0000001)
    if x.__class__ == ndarray:
        N = len(x)
        return array([GETGAMMALN(xi) for xi in x] )#:
    #x += 0.000001
    if x < 5.0:
        y = round( x+0.000001, 2 )
    elif x < 10.0:
        y = round( x+0.000001, 1 )
    else:
        y = int(x+0.000001)
    try:
        return GAMMALN[y]
    except:
        GAMMALN[y] = special.gammaln( y )
        return GAMMALN[y]

def GETLN( x ):
    return log(x+0.0000001)
    if x.__class__ == ndarray:
        return array([GETLN(xi) for xi in x] )
    #x += 0.000001
    if x < 5.0:
        y = round( x + 0.000001, 2 )
    elif x < 10.0:
        y = round( x + 0.000001, 1 )
    else:
        y = int(x+0.000001)
    try:
        return LOG[y]
    except:
        #print y,x
        if y > 0.0:
            LOG[y] = log( y )
        else:
            LOG[y] =  -743.746924741
        return LOG[y]
 
def GaussianDataSummaryLength():  
    return 3

def CreateGaussianDataSummary():
    d = zeros( 3, dtype = float )
    return d

def CreateGaussianDataSummaryEntry( x=None ):
    d = zeros( 3, dtype = float )
    if x is None:  
        return d
    if x != MISSINGVAL:
        d[0] = 1
        d[1] = x
        d[2] = x*x
    return d

def CreateGaussianDataSummaryVec( x=None ):
    d = zeros( 3, dtype = float )
    if x is None:  
        return d
    for xi in x:
        if xi != MISSINGVAL:
            d[0] += 1
            d[1] += xi
            d[2] += xi*xi
    return d

class BiclusteringDataSummary( object ):
    def __init__( self, X, XT, rowU, colV ):
        self.rowDataSummaryMatrix = DataSummaryMatrix( X, colV )
        self.colDataSummaryMatrix = DataSummaryMatrix( XT, rowU )
        self.X = self.rowDataSummaryMatrix.X
        self.XT = self.colDataSummaryMatrix.X

    def ColumnChange( self, oldk, newk, column_xvec, bTranspose, bRemoveOld, bNew ):
        if bTranspose:
            """ change to row mat because cluster id changed in the columns of the orig matrix """
            self.rowDataSummaryMatrix.ColumnClusterChange( oldk, newk, column_xvec, bRemoveOld, bNew )
        else:
            self.colDataSummaryMatrix.ColumnClusterChange( oldk, newk, column_xvec, bRemoveOld, bNew )


    def ColumnPutToEnd( self, k, M ):
        dlen = GaussianDataSummaryLength() 
        X                 = M[:,dlen*k:dlen*k+dlen].copy()
        M[:,dlen*k:-dlen] = M[:,dlen*k+dlen:]
        M[:,-dlen:]       = X

    def ColumnMerge( self, to_k, from_k, M ):
        dlen = GaussianDataSummaryLength() 
        M[:,dlen*to_k:dlen*to_k+dlen] += M[:,dlen*from_k:dlen*from_k+dlen]
        M[:,dlen*from_k:-dlen]         = M[:,dlen*from_k+dlen:]
        M                              = M[:,:-dlen]
        return M

class DataSummaryMatrix( object ):
    def __init__( self, X, V ):
        nRows, nCols = X.shape

        self.DSUM    = []
        self.DSUMTOT = []
        for i in range(nRows):
            dsummarytot  = CreateGaussianDataSummaryVec()
            dlen = GaussianDataSummaryLength()
            rowsummaries = zeros( dlen*len(V), dtype=float )
            for vset,k in zip( V, range(len(V))):
                dsummary     = CreateGaussianDataSummaryVec( X[i][vset] )
                dsummarytot +=  dsummary
                rowsummaries[dlen*k:dlen*k+dlen] = dsummary

            self.DSUM.append( rowsummaries )
            self.DSUMTOT.append( dsummarytot )

        self.DSUM = array( self.DSUM )
        self.DSUMTOT = array( self.DSUMTOT )

        self.X  = []
        for i in range(nRows):
            xrow = []
            for j in range(nCols):
                x = CreateGaussianDataSummaryEntry( X[i][j] )
                xrow.append( array(x,dtype=float) )
            self.X.append( array(xrow,dtype=float))
        self.X  = array( self.X, dtype=float )

    def ColumnClusterChange( self, oldk, newk, column_summary, bRemoveOld, bNew ):
        assert self.DSUM.shape[0] == column_summary.shape[0], 'should be the same time'
        removedSummary = None
        dlen = GaussianDataSummaryLength() 
        if bRemoveOld:
            self.DSUM[:,dlen*oldk:-dlen] = self.DSUM[:,dlen*oldk+dlen:]
            if bNew:
                self.DSUM[:,-dlen:]  = column_summary
            else:
                self.DSUM = self.DSUM[:,:-dlen]
                self.DSUM[:,newk*dlen:newk*dlen+dlen]  += column_summary
        else:
            
            if bNew:
                self.DSUM = concatenate((self.DSUM, zeros((self.DSUM.shape[0],dlen)) ),1)
                self.DSUM[:,oldk*dlen:oldk*dlen+dlen] -= column_summary
                self.DSUM[:,newk*dlen:newk*dlen+dlen] += column_summary
            else:
                self.DSUM[:,oldk*dlen:oldk*dlen+dlen] -= column_summary
                self.DSUM[:,newk*dlen:newk*dlen+dlen] += column_summary


class GaussianDataSummary( object ):
    def __init__( self, x = None ):
        self.d = zeros( 3, dtype = float )
        if x is not None:
            self.Add( x )

    def Add( self, x ):
        if x.__class__ == GaussianDataSummary:
            self.d += x.d
        else:
            self.d += WeaveDataConvert( x )

    def Subtract( self, x ):
        if x.__class__ == GaussianDataSummary:
            self.d -= x.d
        else:
            self.d -= WeaveDataConvert( x )

    def QuickAdd( self, xi ):
        if xi != MISSINGVAL:
            self.d[0] += 1
            self.d[1] += xi
            self.d[2] += xi**2

    def QuickSubtract( self, xi ):
        if xi != MISSINGVAL:
            self.d[0] -= 1
            self.d[1] -= xi
            self.d[2] -= xi**2

    def QuickSubtractWeave( self, xi ):
        X = zeros(3,dtype=float)
        code = r"""
          if (x < 999.0)
          {
             X(0) = 1.0; 
             X(1) = x;
             X(2) = x*x;
          }
          
        """
        vars = ['x','X'] 
        global_dict={'x':float(xi),'X':X} 
        weave.inline(code,vars,global_dict=global_dict, type_converters=weave.converters.blitz)
        self.n -= int(X[0])
        self.xsum -= X[1]
        self.xsumsqrd -= X[2]

    def Verify( self ):
        assert self.n >= 0, 'Counts gone negative!'+str(self.n)
        assert self.xsumsqrd >= 0, 'sum sqrd gone negative!'+str(self.xsumsqrd)

def WeaveDataConvert( x ):
        from scipy import weave
        ndata = len(x)
        X   = zeros(3,dtype=float)
        #Y    = zeros(1,dtype=float)
        code = r"""
          float xx     = 0.0; 
          float XX     = 0.0; 
          float XXSQ   = 0.0; 
          int n        = 0;
          for(int k=0;k<ndim;k++)
          {
              xx   = x(k);
              //xxsq = xsq(k);
              //printf( "x: %f xx: %f bUse: %d\n", xx, xxsq, bUse );
              if (xx < 999.0)
              {
                  n   += 1;
                  XX  += xx;
                  XXSQ  += xx*xx;
                      
              }
          }
          X(0) = (float) n; 
          X(1) = XX;
          X(2) = XXSQ;
        """
        vars = ['x','X','ndim'] 
        global_dict={'x':x,'X':X,'ndim':ndata} 
        weave.inline(code,vars,global_dict=global_dict,\
                       type_converters=weave.converters.blitz)
        return X


def GetDataObservationsBicluster( M, i ):
    data_vec_list = []
    data_total = [0,0.0,0.0]
    x        = M.data.X[i][M.V]
    I        = x != M.data.missingval
    dim      = I.sum()
    xsum     = x[I].sum()
    xsumsqrd = M.data.XX[i][I].sum()
    data_total[0] += dim
    data_total[1] += xsum
    data_total[2] += xsumsqrd
    return data_total



def CategoryConcLogProb( c, params ):
    hyperprior = params[0]
    concs      = params[1]

    logprob = logprobgamma( c, hyperprior.a, hyperprior.b )
    for p in concs:
        logprob += logprobgamma( p, c, 1.0 )
    return logprob

def ConcLogProb( c, params ):
    hyperprior = params[0]
    cnts       = params[1]

    logprob  = logprobgamma( c, hyperprior.conc_a, hyperprior.conc_b )
    logprob += LogPartition( c, cnts )
    return logprob

def ExpectedPrecLogProb( c, params ):
    nu         = params[0]
    hyperprior = params[1]
    prec       = params[2]

    logprob = logprobgamma( c, hyperprior.expprec_a, hyperprior.expprec_b )
    for p in prec:
        logprob += logprobgamma( p, nu/2.0, nu/(2.0*c) )
    return logprob

def DoFLogProb( nu, params ):
    c          = params[0]
    hyperprior = params[1]
    prec       = params[2]
    K = len(prec)
    logprec = log(array(prec,dtype=float))

    logprob = logprobgamma( nu, hyperprior.dof_a, hyperprior.dof_b )
    #for p in prec:
    #    logprob += logprobgamma( p, nu/2.0, nu/(2.0*c) )
    logprob += -K*GETGAMMALN(nu/2.0) + K*nu*GETLN(nu/(c*2.0))/(2.0) + (nu/2.0 - 1.0)*sum(logprec) - nu*sum(prec)/(2.0*c)
    return logprob

def DoFLogProbold( nu, params ):
    c          = params[0]
    hyperprior = params[1]
    prec       = params[2]

    logprob = logprobgamma( nu, hyperprior.dof_a, hyperprior.dof_b )
    for p in prec:
        logprob += logprobgamma( p, nu/2.0, nu/(2.0*c) )
    return logprob

def SliceConc( prior, hyper, weights, val ):
    params   = [hyper, weights]
    Nsamples = 20
    Nburn    = 5
    stepsize = 0.1
    concs,logprob = slice_sample( CategoryConcLogProb, params, prior.conc[val], 0.0, inf, stepsize, Nsamples, 2 )
    return concs[-1],logprob

def SliceDoF( prior, hyperprior, prec ):
    params   = [prior.c, hyperprior, prec]
    Nsamples = 10
    Nburn    = 5
    stepsize = 0.1
    #pdb.set_trace()
    nus,logprob = slice_sample( DoFLogProb, params, prior.nu, 0.0, inf, stepsize, Nsamples, 2 )
    return nus[-1],logprob

def SliceExpectedPrec( prior, hyperprior, prec ):
    params   = [prior.nu, hyperprior, prec]
    Nsamples = 10
    Nburn    = 5
    stepsize = 0.1
    nus,logprob = slice_sample( ExpectedPrecLogProb, params, prior.c, 0.0, inf, stepsize, Nsamples, 2 )
    return nus[-1],logprob

def SliceDirchletProcessConcentration( prior, hyperprior, cnts ):
    params   = [hyperprior, cnts]
    Nsamples = 10
    Nburn    = 5
    stepsize = 0.1
    concs,logprob = slice_sample( ConcLogProb, params, prior.conc, 0.0, inf, stepsize, Nsamples, 2 )
    return concs[-1],logprob

def logprobgamma( x, a, b ):
   if x < 0.00000001:
       L = LOGMIN
       return L
   L = -b*x
   L -= special.gammaln( a )
   L += (a-1.0)*log(x)
   L += a*log(b)
   return L

def logprobnormal( x, mu, vr ):
   logprob  = -0.5*((x-mu)**2)/vr
   logprob += -0.5*(LOG2PI+log(vr))
   return logprob

class SimulationSpecs( object ):
    def __init__( self ):
        self.outdir   = None
        self.nMCMC    = None
        self.nBurnIn  = None

class ClusteringPrior( object ):
    def __init__( self ):
        self.bUseRowBias      = 0 # use row cluster parameters
        self.bUseColumnBias   = 0 # use column cluster parameters
        self.bUseBicluster    = 1 # use block-constant bicluster parameters
        self.rowprior         = None # concentration parameter for the row Pitman-Yor process
        self.colprior         = None # discount parameter for the column Pitman-Yor process

class BiclusteringPrior( object ):
    def __init__( self ):
        self.biclusterType     = None
        self.bicluster_params  = None
        self.rowcluster_params = None
        self.colcluster_params = None
        self.bic_hyper         = None
        self.row_hyper         = None
        self.col_hyper         = None

def BuildModel( modelPrior, clusteringPrior, data=None, names= None ):
    if data is not None:
        clusteringPrior.rowPrior.N = data.nRows
        clusteringPrior.colPrior.N = data.nCols
    if modelPrior.biclusterType     == 'NonConjugateGaussianBicluster':
      return BiclusteringNonConjugateGaussianModel( clusteringPrior, modelPrior, data, names )
    elif modelPrior.biclusterType     == 'ConjugateGaussianBicluster':
      return BiclusteringConjugateGaussianModel( clusteringPrior, modelPrior, data, names )
    #return BiclusteringMultinomialModel( clusteringPrior, modelPrior, data, names )

"""
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CLUSTER PRIOR CLASS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
class ClusterPrior( object ):
    def __init__( self, conc = 1.0, discount = 0.0, N = 0 ):
        """ 
            size: nbr rows/cols
            conc: concentration for rows
            discount: pitman-your discount for rows
        """
        self.N        = N
        self.conc     = conc
        self.discount = discount

    def Sample( self ):
        return PitmanYor( self.N, self.conc, self.discount )

class ClusterHyperPrior( object ):
    def __init__( self, conc_a = 5.0, conc_b = 1.0 ):
        self.conc_a        = conc_a
        self.conc_b        = conc_b

""" 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    BICLUSTERING CLASS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
class BiclusteringModel( object ):
    def __init__(self, clusteringPrior, modelPrior, data = None, names = None):
        # self.BG       = BG # known interactions from biogrid
        #         self.GO       = GO # known interactions from go annotations
        self.cPrior   = clusteringPrior
        self.rowPrior = clusteringPrior.rowPrior
        self.colPrior = clusteringPrior.colPrior
        self.mPrior   = modelPrior
        self.data     = data
        self.dim      = self.data.dim
        self.names    = names
        self.nRows,self.nCols = self.data.X.shape
        self.U = None
        self.V = None
        self.Biclusters     = None
        self.Rowclusters    = None
        self.Columnclusters = None
        self.rowU           = None
        self.colV           = None

        self.nNewClustersFromGibbs = 0
        # algorithm variables
        self.bTranspose = False
        self.bAllowNew  = True
        self.newbiclusters = []
        self.newrowcluster = None
        self.newcolcluster = None

        self.nTestImputeCalls = 0
        # logging
        self.LogProbGibbs = 0.0
        self.LogProbData  = 0.0
        self.rowNeighbours = []
        for i in range( self.nRows ):
            self.rowNeighbours.append( zeros( self.nRows, dtype = float ) )
        self.rowNeighbours = array( self.rowNeighbours)
        self.rowClustering = self.rowNeighbours.copy()
        self.colNeighbours = []
        for i in range( self.nCols ):
            self.colNeighbours.append( zeros( self.nCols, dtype = float ) )
        self.colNeighbours = array( self.colNeighbours)
        self.colClustering = self.colNeighbours.copy()
        
        self.row_conc = []
        self.col_conc = []

    """ 
        -----------------------------------
        Init: 
           1) generate initial row (U) and column (V) partitions.
           2) initialize parameters for biclusters, rowclusters, colclusters
        -----------------------------------
    """
    def Init(self):
        dlen = GaussianDataSummaryLength()
        self.InitHyperParameters()
        # generate random row and column paritions
        self.U,self.rowCnts = self.rowPrior.Sample()
        self.V,self.colCnts = self.colPrior.Sample()
        self.K              = len(self.rowCnts)
        self.L              = len(self.colCnts)
        self.rowU = []
        self.colV = []
        for u in range( self.K ):
            I = find( self.U == u ).tolist()
            self.rowU.append(I)
        for v in range( self.L ):
            I = find( self.V == v ).tolist()
            self.colV.append(I)

        #INITGAMMALN( self.data.nEntries )
        self.useU = self.rowU

        self.biclusteringDataSummary = BiclusteringDataSummary( self.data.X, self.data.XT, self.rowU, self.colV )

        self.row_name_to_id_dict = {}
        self.col_name_to_id_dict = {}
        
        if self.names is not None:
            for i,nm in zip( range( self.nRows ), self.names.rownames ):
                self.row_name_to_id_dict[nm] = i
            for i,nm in zip( range( self.nCols ), self.names.colnames ):
                self.col_name_to_id_dict[nm] = i
        else:
            for i in range( self.nRows ):
                self.row_name_to_id_dict[i] = i
            for i in range( self.nCols ):
                self.col_name_to_id_dict[i] = i


        """ 
           BICLUSTERS
        """
        self.bicDataSummary = []
        self.bicParameters  = []
        #self.rowParameters  = []
        #self.colParameters  = []
        for I in self.rowU:
            rowsummary    = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[I].sum(0)
            rowsummarytot = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[I].sum(0)
            bicrowparams,rowparams  = self.InitParams( rowsummary, rowsummarytot )
            self.bicDataSummary.append( rowsummary )
            self.bicParameters.append( bicrowparams )
            #if rowparams is not None:
            #    self.rowParameters.append( rowparams )
        self.bicDataSummary = array( self.bicDataSummary )
        self.bicParameters  = array( self.bicParameters )
        #if rowparams is not None:
        #    self.rowParameters = array( rowparams )
        #for J in self.colV:
        #    colsummarytot    = self.biclusteringDataSummary.colDataSummaryMatrix.DSUMTOT[J].sum(0)
        #    colparams  = self.InitColParams( colsummarytot )
        #    #if colparams is not None:
        #    #    self.colParameters.append( colparams )
        #if colparams is not None:
        #    self.colParameters.append( colparams )
        
        #self.Verify("INIT")
        self.UpdateParameters()
    """ 
        -----------------------------------
        Show: 
           1) Visualize the partitions for all cluster types.
        -----------------------------------
    """
    def Show( self, outfile, iter ):
        
        order_rows,order_cols = self.ShowBiclusters( 1 )
        return
        #self.ShowRowclusters( 3 )
        #self.ShowColclusters( 4 )
        self.ShowNeigbours( 5, self.rowNeighbours, self.rowU )
        self.ShowNeigbours( 6, self.colNeighbours, self.colV )
        self.PrintClusters(outfile)
        self.PrintDotty( outfile )
        #self.PrintSif( outfile )
        self.ShowHyperparameters(12) 
        #self.known_complexes.View( order_rows, self.rowCnts, 15, bRows = True )
        #self.known_complexes.View( order_cols, self.colCnts, 16, bRows = False )
        self.known_complexes.ViewScore( 15 )
        self.known_complexes.ComplexScore( self.rowNeighbours, self.colNeighbours, iter+1 )

    def ShowBiclusters( self, fig ):
        X = ones( (self.nRows, self.nCols), dtype=float )
        self.rowPerm = []
        self.colPerm = []
        cntsU = []
        cntsV = []
        for I in self.rowU:
            self.rowPerm.extend( I )
            cntsU.append( len( I ) )
        for J in self.colV:
            self.colPerm.extend( J )
            cntsV.append( len( J ) )
        figure(fig,figsize = (8,8))
        clf()
        ioff()
        XX = self.data.X.copy()
        for i,XJ in zip( range( self.nRows ), self.data.missing.X ):
            J = find( XJ == True )
            for j in J:
                XX[i][j] = 0.0
        #print "MEAN, STD, PREC:", mean(XX),std(XX), 1.0/var(XX)
        Y = matshow( XX[self.rowPerm,:][:,self.colPerm] ,fignum=fig,cmap=cm.jet,aspect='auto' )
        
        try:
            colorbar( )
        except:
            pass
        # add partition lines to image
        self.AddLines( cntsU, cntsV, fignum=fig )
        ion() 
        draw()
        return self.rowPerm,self.colPerm

    def PrintDotty( self, outdir ):
        self.row_dotty = DotGeneGraph( fname=outdir + "_row.dot" )
        self.PrintNeighbourhoodDotty( self.row_dotty, self.rowNeighbours, self.names.rownames )
        self.col_dotty = DotGeneGraph( fname=outdir + "_col.dot" )
        self.PrintNeighbourhoodDotty( self.col_dotty, self.colNeighbours, self.names.colnames, bRows = False )
        #self.row_dotty.PrintToFile()
        #self.col_dotty.PrintToFile()

        #self.PrintToSif( self.row_dotty )
        #self.PrintToSif( self.col_dotty )
        self.PrintToTab( self.row_dotty )
        self.PrintToTab( self.col_dotty, bTranspose = True )

    def PrintToTab( self, dottyobject, bTranspose = False ):
        #pdb.set_trace()
        fname = dottyobject.fname
        fname = fname.replace('.dot','.tab')
        fid = open( fname, 'w+' )
        id_to_name = {}
        for node in dottyobject.nodes:
            id_to_name[ node.id] = node.label
        if bTranspose is False:
            self.data_score = []
        fid.write("Source\tInteraction\tTarget\tEdgeW\tInteractionW\tDataW\tInteractionS\tInteractionAbs\n")
        for edge in dottyobject.edges:
             nm1 = id_to_name[edge.a]
             nm2 = id_to_name[edge.b]
             weight = edge.weight
             if bTranspose:
                 j,cj = self.GetColId( nm1 )
                 i,ci = self.GetRowId( nm2 )
             else:
                 i,ci = self.GetRowId( nm1 )
                 j,cj = self.GetColId( nm2 )
             bUseDataWeight = False
             if j != -1 and i != -1:
                 if bTranspose:
                     dataweight = self.data.XT[j][i]
                 else:
                     dataweight = self.data.X[i][j]
                 if dataweight == 999.0:
                     dataweight = 0.0
                 else: 
                     bUseDataWeight = True
             else:
                 dataweight = 0.0 
             if cj != -1 and ci != -1:
                 bicweight = self.Biclusters[ci][cj].GetShowMean() + self.rowclusters[ci].GetShowMean() + self.colclusters[cj].GetShowMean()
                 if bUseDataWeight and bTranspose is False:
                     self.data_score.append([dataweight, bicweight])
             else:
                 bicweight = 0.0
             bicsign = sign(bicweight)
             bicabs  = abs( bicweight )
             fid.write( "%s\t%s\tto\t%.2f\t%.2f\t%.2f\t%d\t%.2f\n"%(nm1,nm2,weight,bicweight,dataweight,int(bicsign), bicabs) )
        fid.close()

    def GetRowId( self, nm ):
        try:
            i = self.row_name_to_id_dict[ nm ]
            return i,self.U[i]
        except:
            return -1,-1

    def GetColId( self, nm ):
        try:
            i = self.col_name_to_id_dict[ nm ]
            return i,self.V[i]
        except:
            return -1,-1

    def PrintToSif( self, dottyobject ):
        #pdb.set_trace()
        fname = dottyobject.fname
        fname = fname.replace('.dot','.sif')
        fid = open( fname, 'w+' )
        id_to_name = {}
        for node in dottyobject.nodes:
            id_to_name[ node.id] = node.label
        for edge in dottyobject.edges:
             fid.write( "%s\tto\t%s\n"%(id_to_name[edge.a],id_to_name[edge.b] ) )
        fid.close()

        fname = dottyobject.fname
        fname = fname.replace('.dot','.noa')
        fid = open( fname, 'w+' )
        fid.write("NodeInComplex\n")
        for node in dottyobject.nodes:
             if node.label.find( '[' ) > 0:
                 nodetype = 'hasComplex'
             else:
                 nodetype = 'noComplex'
             fid.write( "%s\t=\t%s\n"%(node.label, nodetype) )
        fid.close()

        fname = dottyobject.fname
        fname = fname.replace('.dot','.eda')
        fid = open( fname, 'w+' )
        fid.write("EdgeWeight\n")

        for edge in dottyobject.edges:
             nm1 = id_to_name[edge.a]
             nm2 = id_to_name[edge.b]
             weight = edge.weight
             fid.write( "%s\t(to)\t%s\t=\t%.2f\n"%(nm1,nm2,weight ) )
        fid.close()

        fname = dottyobject.fname
        fname = fname.replace('.dot','_interaction.eda')
        fid = open( fname, 'w+' )
        fid.write("InteractionSign\n")
        for ci,nm1 in zip( self.U, self.names.rownames ): 
           for cj,nm2 in zip( self.V, self.names.colnames ):
               weight = self.Biclusters[ci][cj].GetShowMean() + self.rowclusters[ci].GetShowMean() + self.colclusters[cj].GetShowMean()
               fid.write( "%s\t(to)\t%s\t=\t%.2f\n"%(nm1,nm2,weight ) )
        fid.close()
        #fid2.close()

    def PrintNeighbourhoodDotty( self, dotty, N, names, bRows = True ):
        dicti = self.known_complexes.name_to_comp_dict
        if bRows:
            cnts = self.known_complexes.row_complex_count
        else:
            cnts = self.known_complexes.col_complex_count
        for i, nm, complex_count in zip( range( N.shape[0] ), names, cnts ):
            if complex_count > 0:
                altnm = nm #+ ' ' + str(dicti[nm]) 
                dotty.nodes.append( DotGeneNode( id = i, label = altnm, shape = 'box' ) )
            else:
                dotty.nodes.append( DotGeneNode( id = i, label = nm ) )
        tabulist = {}
        for i, rwnm in zip( range( N.shape[0] ), N ):
            I,L,W = self.GetIndicesToPutInGraph( rwnm )
            #I = (-rwnm).argsort()
            #pdb.set_trace()
            for j, length, weight in zip( I, L, W ):
                if self.InTabu( tabulist, i, j ) is False and i != j:
                    dotty.edges.append( DotGeneEdge( a = i, b = j, length = length, weight = weight ) )
                    try:
                        tabulist[i].append(j)
                    except:
                        tabulist[i] = [j]
                    try:
                        tabulist[j].append(i)
                    except:
                        tabulist[j] = [i]

    def GetIndicesToPutInGraph( self, row ): 
        threshold = 0.5
        newrow = array( row, dtype=float ) / row.max()
        newrow *= -1.0
        J = newrow.argsort()
        newrow *= -1.0
        
        I = find( newrow > threshold ) #I = (-newrow).argsort()
        #I = (-newrow[I]).argsort()
        #print I, newrow
        nuse  = min( 10, len(I) )
    
        if nuse == 0:
            return [],0.0,0.0

        I = J[0:nuse]
        W = newrow[I] / newrow[I].max()
        L = 0.1 / W
        #print J[I], newrow[J[I]]
        #pdb.set_trace()
        return I,L,W
    def InTabu( self, tabulist, i, j ):
        if len( tabulist ) == 0 :
            return False
        try:
            if j in tabulist[i]:
                return True
            else:
                return False
        except:
            return False


    def ShowNeigbours( self, fig, N, useU ):
        figure(fig,figsize = (2,2))
        clf()
        ioff()
        U = []
        # for each cluster, get block constant value
        for J in useU:
            U.extend( J )
        matshow( N[U,:][:,U],fignum=fig,cmap=cm.hot )
        try:
            colorbar( )
        except:
            pass
        ion() 
        draw()


    def ShowHyperparameters( self, fig ):
        figure(fig,figsize = (3,3))
        clf()
        ioff()
        if len( self.row_conc ) > 0:
            plot( self.row_conc, 'g' )
        if len( self.col_conc ) > 0:
            plot( self.col_conc, 'r' )
        ion() 
        draw()

    def AddLines( self, cntsU, cntsV, fignum ):
       x = 0
       y = 0
       I = sum( cntsU )
       J = sum( cntsV )
       for n in cntsU:
           y += n
           hlines( y-0.5, 0, J-1, 'g' )
       for n in cntsV:
           x += n
           vlines( x-0.5, 0, I-1, 'g' )

    """ 
        -----------------------------------
        Transpose: 
           Alternate working on rows/columns (change all clusters).
        -----------------------------------
    """
    def Transpose( self ):
        if self.bTranspose:
            """ work on COLUMNS """
            self.bTranspose = False
            self.useU = self.rowU
        else:
            """ work on ROWS """
            self.bTranspose = True
            self.useU = self.colV
        #self.ReshapeBiclusters()
        #self.bicDataSummary = self.bicDataSummary.T
        #self.bicParameters  = self.bicParameters.T

    def GetConc(self):
        if self.bTranspose:
            return self.colPrior.conc
        else:
            return self.rowPrior.conc

    def GetDiscount(self):
        if self.bTranspose:
            return self.colPrior.discount
        else:
            return self.rowPrior.discount

    def GetNbrObjects(self):
        if self.bTranspose:
            return self.nCols
        else:
            return self.nRows

    def GetNbrClusters(self):
        if self.bTranspose:
            return self.L
        else:
            return self.K

    """ 
        -----------------------------------
        RunGibbs: 
           Run a single Gibbs sweep over rows and columns.
        -----------------------------------
    """
    def RunGibbs( self, t ):
        
        for i in range(self.simspecs.nSplitMergeMove):
            self.SplitMergeSide(t)
        for i in range(self.simspecs.nAdjustedMove):
            self.AdjustedNonconjugateSide()
        self.GibbsSide()
        #self.Verify("PRE SPLIT-MERGE")
        
        #return
        self.UpdateParameters()
        self.Transpose()
        #self.Verify("PRE SPLIT-MERGE")

        for i in range(self.simspecs.nSplitMergeMove):
            self.SplitMergeSide(t)
        for i in range(self.simspecs.nAdjustedMove):
            self.AdjustedNonconjugateSide()
        #self.Verify("PRE SPLIT-MERGE")
        self.GibbsSide()
        #self.Verify("PRE SPLIT-MERGE")
        #print "UPDATing PARAMS"
        self.UpdateParameters()
        #self.UpdateHyperParameters()
        #self.Verify("PRE SPLIT-MERGE")
        self.Transpose()
        #self.Verify("PRE SPLIT-MERGE")
        self.ImputeTestValues()


    """ 
        -----------------------------------
        GibbsSide: 
           Run a single Gibbs sweep either row or column indicators.
        -----------------------------------
    """
    def GibbsSide( self ):
        #assert any(array(self.U)==-1) is False, "U had undefined values"
        #assert any(array(self.V)==-1) is False, "V had undefined values"
        
        self.LogProbGibbs = 0.0
        self.LogProbData  = 0.0
        N = self.GetNbrObjects()
        P = random.permutation( N )
        #self.Verify("PRE-GIBBS")
        nChanges = 0
        for i in P:
            #print "GIBBS on ",i
            bColumnChanged = self.GibbsUpdate( i )  
            if bColumnChanged:
                nChanges += 1
            #self.Verify("GIBBS")
        print nChanges,' of ',N,' changes made.'
        #self.Verify("POST-GIBBS")



    """ 
        -----------------------------------
        GibbsUpdate(i): 
           Update the cluster indicator for the ith data-vector.
        -----------------------------------
    """
    def GibbsUpdate( self, i ):
        # remove from current cluster
        oldk = self.GetClusterId( i )
        
        if self.bTranspose:
            data_summary_list  = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[i]
        else:
            data_summary_list  = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[i]
        bRemoveOld = self.RemoveFromClusters( i, oldk, data_summary_list )
        #print "B"
        #if bRemoveOld: 
        #    print "REMOVED OLD"

        # gather log probs of prior and likelihoods
        logprior = self.GetPriorLogProb()
        K        = self.GetNbrClusters()
        if self.bAllowNew:
            logcomps  = zeros( self.GetNbrClusters()+1, dtype=float )
        else:
            logcomps  = zeros( self.GetNbrClusters(), dtype=float )
        for cid in range(K):
            logcomps[cid] = self.GetLogProbCluster( cid, data_summary_list )
        if self.bAllowNew:
            nEstimate = self.simspecs.nSamplesForNew
            logprobnew = zeros( nEstimate, dtype = float )
            #pdb.set_trace()
            for ne in range( nEstimate ):
                self.ResetNewClusters()
                logprobnew[ne] = self.GetLogProbNewCluster( data_summary_list )
                
            if nEstimate > 0:
                logcomps[-1] = LogSum( logprobnew ) - log(nEstimate)
            else:
                logcomps[-1] = logprobnew
 
        #print logcomps
        # prob is proportional to prob under prior times data likelihood
        loglik = logprior + logcomps
        ls     = LogSum( loglik )
        # normalize in log
        lr     = loglik - ls
        #print logprior, logcomps, loglik,exp(lr)
        # sample from normalized log vector
        if self.simspecs.bMaximize:
            newk   = argmax( lr )
        else:
            newk   = LogRnd( lr )
        # gather stats
        self.LogProbGibbs    += lr[newk]
        self.LogProbData     += ls
        #self.LogProbDataAcc[i]  += logcomps[newk]

        # insert into new cluster
        bNew = False
        if newk == self.GetNbrClusters():
            bNew = True
            #print "NEW CLUSTER"
            #pdb.set_trace()
        self.AddToClusters( i, newk, data_summary_list  )

        bChange = False
        if newk != oldk or bNew or bRemoveOld:
            """ a cluster change took place """
            bChange = True
            if self.bTranspose:
                self.biclusteringDataSummary.ColumnChange( oldk, newk, self.biclusteringDataSummary.XT[i], self.bTranspose, bRemoveOld, bNew )
            else:
                self.biclusteringDataSummary.ColumnChange( oldk, newk, self.biclusteringDataSummary.X[i],  self.bTranspose, bRemoveOld, bNew )
        assert self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM.shape[1]/DLEN == self.L, 'bad'
        assert self.biclusteringDataSummary.colDataSummaryMatrix.DSUM.shape[1]/DLEN == self.K, 'bad'

        if bNew:
            self.UpdateSpecificParameters(newk)
            self.nNewClustersFromGibbs += 1
        pvec = exp( lr )
        if self.bTranspose:
            self.AddProbsToNeighbours( i, pvec, self.colNeighbours, self.colV )
            self.AddProbsToClustering( i, self.colClustering, self.colV[newk] )
        else:
            self.AddProbsToNeighbours( i, pvec, self.rowNeighbours, self.rowU )
            self.AddProbsToClustering( i, self.rowClustering, self.rowU[newk] )
        return bChange

    def AddProbsToNeighbours( self, i, pvec, neighbours, cluster_ids ):
        for c_ids, pval in zip( cluster_ids, pvec ):
            for j in c_ids:
                neighbours[i][j] += pval
    def AddProbsToClustering( self, i, neighbours, c_ids ):
        for j in c_ids:
            neighbours[i][j] += 1.0

    def UpdateParameters(self):
        """
           UPDATE BICLUSTER PARAMETERS
        """
        for rowdsum, rowparams in zip( self.bicDataSummary, self.bicParameters ):
            logPrior, logPost = self.UpdateParamsVector( rowdsum, rowparams )

    def InitHyperParameters(self):
        self.cPrior.colPrior.conc = random.gamma( self.cPrior.col_hyper.conc_a, 1.0 / self.cPrior.col_hyper.conc_b )
        self.cPrior.rowPrior.conc = random.gamma( self.cPrior.row_hyper.conc_a, 1.0 / self.cPrior.row_hyper.conc_b )
        self.row_conc.append(self.cPrior.rowPrior.conc)
        self.col_conc.append(self.cPrior.colPrior.conc)

    def UpdateHyperParameters(self):
        self.cPrior.rowPrior.conc, lp = SliceDirchletProcessConcentration( self.cPrior.rowPrior, self.cPrior.row_hyper, array(self.rowCnts) )
        self.cPrior.colPrior.conc, lp = SliceDirchletProcessConcentration( self.cPrior.colPrior, self.cPrior.col_hyper, array(self.colCnts) )
        self.row_conc.append(self.cPrior.rowPrior.conc)
        self.col_conc.append(self.cPrior.colPrior.conc)

    def UpdateSpecificParameters(self, ci, cj = None ):
        self.RestrictedLogProbParamsPrior = 0.0
        self.RestrictedLogProbParamsPost  = 0.0
        if cj is None:
            I = [ci]
        else:
            if cj == self.GetNbrClusters():
                # we have divered, only update ci
                I = [ci]
            else:
                I = [ci,cj]

            """
               UPDATE BICLUSTER PARAMETERS
            """
        for rowdsum, rowparams in zip( self.bicDataSummary, self.bicParameters ):
            logPrior, logPost = self.UpdateParamsVector( rowdsum, rowparams, I )
            self.RestrictedLogProbParamsPrior += logPrior
            self.RestrictedLogProbParamsPost  += logPost
        
    def ImputeTestValues( self ):
        self.testerror = 0.0
        self.testerroracc = 0.0
        if self.data.TPairs is None:
            return

        if self.nTestImputeCalls == 0:
            self.TESTIMPUTE    = zeros( self.data.nTest, dtype = float )
            self.TESTIMPUTEACC = zeros( self.data.nTest, dtype = float )
        for n,tpair,truex in zip( range(self.data.nTest), self.data.TPairs, self.data.TestX ):
            i = tpair[0]
            j = tpair[1]
            u = self.U[i]
            v = self.V[j]
            
            imputedValue = self.SampleDatum( u, v )
            self.TESTIMPUTE[n]    = imputedValue
            self.TESTIMPUTEACC[n] = (self.nTestImputeCalls * self.TESTIMPUTEACC[n] + imputedValue)/float(self.nTestImputeCalls+1)
            self.testerror    += (imputedValue - truex )**2
            self.testerroracc += (self.TESTIMPUTEACC[n] - truex )**2
        self.nTestImputeCalls += 1
        self.testerror /= float( self.data.nTest )
        self.testerroracc /= float( self.data.nTest )



    """ 
        -----------------------------------
        GetClusterId( i ): 
           Which cluster label does the ith data-vector belong to?
        -----------------------------------
    """
    def GetClusterId( self, i ):
        if self.bTranspose:
            return self.V[i]
        else:
            return self.U[i]

    """ 
        -----------------------------------
        AddToClusters( i, k ): 
           Insert the ith data-vector (and indicator) into the kth cluster.
        -----------------------------------
    """
    def AddToClusters( self, i, k, data_summary_list ):
        dlen = GaussianDataSummaryLength()
        if self.bTranspose:  
            if k == self.L: 
                # add new biclusters
                self.bicDataSummary = concatenate((self.bicDataSummary, zeros((1,self.bicDataSummary.shape[1])) ),0)
                self.bicParameters  = concatenate((self.bicParameters, zeros((1,self.bicParameters.shape[1])) ),0)
                self.L += 1
                self.colCnts.append(0)
                self.colV.append(self.newU)
                self.bicParameters[k]  += self.newBicParameters

            self.AddToSet( self.colV[k], i )
            self.bicDataSummary[k] += data_summary_list
            self.V[i] = k
            self.colCnts[k] += 1
        else: 
            if k == self.K: 
                #Log( "\t\tAdding new ROW cluster")
                # add new biclusters
                self.bicDataSummary = concatenate((self.bicDataSummary, zeros((1,self.bicDataSummary.shape[1])) ),0)
                self.bicParameters  = concatenate((self.bicParameters, zeros((1,self.bicParameters.shape[1])) ),0)

                self.K += 1
                self.rowCnts.append(0)
                self.rowU.append(self.newU)
                self.bicParameters[k]  += self.newBicParameters

            self.bicDataSummary[k] += data_summary_list
            self.AddToSet( self.rowU[k], i )
            self.U[i] = k
            self.rowCnts[k] += 1
        self.ResetNewClusters()

    def ResetNewClusters(self):
        self.newBicDataSummary = []
        self.newBicParameters  = []

    """ 
        -----------------------------------
        RemoveFromClusters( i, k ): 
           Remove the ith data-vector (and indicator) from the kth cluster.
        -----------------------------------
    """
    def RemoveFromClusters( self, i, k, data_summary_list ):
        bRemoveOld = False
        assert k != -1, "remove from undefined cluster!"

        self.bicDataSummary[k] -= data_summary_list
        #print self.bicDataSummary[k]
        if self.bTranspose:  
            self.V[i] = -1
            self.colCnts[k] -= 1 
            self.RemoveFromSet( self.colV[k], i )
            if self.colCnts[k] == 0:
                #print "REMOVING COLUMN"
                self.newBicParameters = self.bicParameters[-1].copy()
                self.V[self.V>k]-=1
                self.L -= 1
                bRemoveOld = True
                self.colCnts.pop(k)
                self.colV.pop(k)
                #print self.bicDataSummary[k], data_summary_list
                #print self.bicDataSummary.sum(0), k
                self.bicDataSummary[k:-1,:] = self.bicDataSummary[k+1:,:]
                self.bicDataSummary = self.bicDataSummary[:-1,:]
                #print self.bicDataSummary.sum(0), k
                self.bicParameters[k:-1,:] = self.bicParameters[k+1:,:]
                self.bicParameters = self.bicParameters[:-1,:]
            assert self.L == len(self.colCnts), "what?"
            assert self.bicDataSummary.shape[0] == self.L, "what?"
        else:   
            self.RemoveFromSet( self.rowU[k], i )       
            self.U[i]       = -1
            self.rowCnts[k] -= 1

            if self.rowCnts[k] == 0:
                #print "REMOVING ROW"
                #pdb.set_trace()
                # print self.bicDataSummary.sum(0)
                self.newBicParameters = self.bicParameters[-1].copy()
                self.U[self.U>k]-=1
                self.rowU.pop(k)
                self.rowCnts.pop(k)
                self.K -= 1
                bRemoveOld = True
                #print self.bicDataSummary[k], data_summary_list
                #print self.bicDataSummary.sum(0), k
                self.bicDataSummary[k:-1,:] = self.bicDataSummary[k+1:,:]
                self.bicDataSummary = self.bicDataSummary[:-1,:]
                #print self.bicDataSummary.sum(0), k
                self.bicParameters[k:-1,:] = self.bicParameters[k+1:,:]
                self.bicParameters = self.bicParameters[:-1,:]
            assert self.K == len(self.rowCnts), "what?"
            assert self.bicDataSummary.shape[0] == self.K, "what?"
        return bRemoveOld

    def GetCount( self, k ):
        if self.bTranspose:
            return self.colCnts[k]
        else:
            return self.rowCnts[k]

    """ 
        -----------------------------------
        GetPriorLogProb: 
           Get vector of log probs for each possible cluster.
        -----------------------------------
    """
    def GetPriorLogProb( self ):
        K        = self.GetNbrClusters()
        if self.bAllowNew:
            logprior = zeros( K + 1, dtype=float )
            logprior[-1] = log( self.GetConc() + K*self.GetDiscount() )
        else:
            logprior = zeros( K, dtype=float )
        if self.bTranspose:
            cnts = self.colCnts
        else:
            cnts = self.rowCnts

        for i,N in zip( range(K), cnts ):
            logprior[i] = log(N)
        return logprior

    """ 
        -----------------------------------
        GetLogProbCluster( i, k ): 
           Get log prob of the ith data-vector under the kth cluster.
        -----------------------------------
    """
    def GetLogProbCluster( self, k, d_list ):
        return self.GetLogProbClusterVec( d_list, self.bicParameters[k], self.bicDataSummary[k] )

    """ 
        -----------------------------------
        GetLogProbNewCluster(i): 
           New clusters are separate: get log prob ith data-vector under this list of biclusters + row/col cluster.
        -----------------------------------
    """
    def GetLogProbNewCluster( self, d_list ):
        if len( self.newBicParameters ) == 0:
            self.GenerateNewClusters( )
        return self.GetLogProbClusterVec( d_list, self.newBicParameters  )

    """ 
        -----------------------------------
        GenerateNewClusters: 
           Randomly generate a new row/col cluster, including biclusters for an entire row/column.
        -----------------------------------
    """
    def GenerateNewClusters( self ):
        self.newBicParameters = self.GenerateRandomBiclusterRow(self.mPrior.bicluster_params )
        self.newU = []

    def AddToSet( self, SET, ID ): 
        if SET is not None:
            assert ID not in SET, "ID " + str(ID) + " already in set: " + str(SET)
        else:
            SET = []
        SET.append( ID )

    def RemoveFromSet( self, SET, ID ):
        assert ID in SET, "ID " + str(ID) + " already not in set: " + str(SET)
        SET.remove( ID )

    """ 
        -----------------------------------
        GatherNeighbours(i): 
           (For keeping track of neighbourhood graph. ) 
           Add current clusters to current neighbourhood graph.
        -----------------------------------
    """
    def GatherNeighbours(self):
        if self.bTranspose:
            self.AddToNeighbours( self.colV, self.colNeighbours )
        else:
            self.AddToNeighbours( self.rowU, self.rowNeighbours )

    """ 
        -----------------------------------
        AddToNeighbours( C, N): 
           (For keeping track of neighbourhood graph.  )
           Add clusters (C) to neighbourhood graph (N).
        -----------------------------------
    """
    def AddToNeighbours( self, useU, N ):
        for I in useU:
            for i in I:
                for j in I:
                    N[i][j] += 1

    def GetClusterIdsForSplitMerge( self, N ):
        i = random.randint( N )
        j = random.randint( N )
        while j == i:
            j = random.randint( N )
        ci = self.GetClusterId( i )
        cj = self.GetClusterId( j )

        return ci,cj
        
        i = random.randint( N )
        j = random.randint( N )
        while j == i:
            j = random.randint( N )
        ci = self.GetClusterId( i )
        cj = self.GetClusterId( j )
        nTries = 0
        nMaxTries = 10
        if random.rand() < 0.5:  
            # split            
            i = random.randint( N )
            ci = self.GetClusterId( i )
            cj = ci
            while self.GetCount(ci) < 2:
              i = random.randint( N )
              ci = self.GetClusterId( i )
              nTries+=1
              if nTries > nMaxTries:
                  while ci == cj:
                      i = random.randint( N )
                      j = random.randint( N )
                      while j == i:
                          j = random.randint( N )
                      ci = self.GetClusterId( i )
                      cj = self.GetClusterId( j )
                      nTries+=1
                      if nTries > nMaxTries:
                          return ci,cj
        else:
            if self.GetNbrClusters() == 1: 
                return 0,0
            while ci == cj:
              i = random.randint( N )
              j = random.randint( N )
              while j == i:
                  j = random.randint( N )
              ci = self.GetClusterId( i )
              cj = self.GetClusterId( j )
              nTries+=1
              if nTries > nMaxTries:
                  return ci,cj
        return ci,cj

    def SplitMergeSide( self, t ):
        #print "begin split/merge"
        #self.Verify("PRE SPLIT-MERGE")
        self.nRestricted = 3
        N = self.GetNbrObjects()
        ci,cj = self.GetClusterIdsForSplitMerge( N )
        if t < self.simspecs.nBurnIn:
            while ci != cj:
                ci,cj = self.GetClusterIdsForSplitMerge( N )
        bChanged = False
        if ci == cj:
            return self.SplitMove( ci, cj, N )
        else: 
            print "trying merge"
            origci = ci
            origcj = cj
            ci,cj = self.GetMergeCandidates(ci)
            if ci != cj:
                return self.MergeMove( ci, cj, N )
            else:
                return self.MergeMove( origci, origcj, N )

    def ResetSplitMergeRestrictedLogProb( self ):
        self.RestrictedLogProbData        = 0.0
        self.RestrictedLogProbParamsPost  = 0.0
        self.RestrictedLogProbParamsPrior = 0.0

    def SplitMove( self, ci, cj, N ):
        #print "begin split"
        # propose splitting ci into ci and cj (now K+1)
        self.ResetSplitMergeRestrictedLogProb()
        cnts,oldU,I = self.GetSplitCountsAndIndicators( ci )
        # keep track of current states
        self.BackupClusters( ci )
        lastk = self.GetNbrClusters()-1
        """ compute reverse probabilities """
        for t in range(1):
            self.UpdateSpecificParameters( ci )
            bDiverged, usable_c = self.RestrictedGibbsSide( I, ci )
            #print "DIVERGED"
            #self.Verify()
        logaccb = - log(self.GetConc()) + GETGAMMALN(cnts[ci])  \
                  - self.RestrictedLogProbParamsPost \
                  + self.RestrictedLogProbParamsPrior \
                  + self.RestrictedLogProbData

        if isnan( logaccb ):
            pdb.set_trace()
        self.ResetSplitMergeRestrictedLogProb()
        """ compute forward probabilities """
        self.ResetNewClusters()
        self.RandomlySplit( ci )
        for t in range(self.nRestricted):
            self.UpdateSpecificParameters( ci )
            bDiverged,usable_c  = self.RestrictedGibbsSide( I, ci, self.GetNbrClusters()-1 )

        logaccf = GETGAMMALN(cnts[ci]) + GETGAMMALN(cnts[lastk]) \
                 - self.RestrictedLogProbParamsPost \
                 + self.RestrictedLogProbParamsPrior \
                 + self.RestrictedLogProbData

        #print "SPLIT F: ",logaccf," B: ",logaccb, " dif: ",logaccf-logaccb,bDiverged
        if isnan( logaccf ):
            pdb.set_trace()
        if random.rand() > min( exp( logaccf-logaccb ), 1.0 ) or bDiverged == True:
            self.RevertFromSplit( ci )
        else:
            print "+++ SPLIT ACCEPTED +++"
            #if self.bTranspose:
            #    print ci, self.L, self.colCnts
            #else:
            #    print ci, self.K, self.rowCnts
            bChangedOnce = False
            for i in I:
                if self.bTranspose:
                    c_i = self.V[i]  
                    old_ci = oldU[i]
                    if c_i != ci:
                        if bChangedOnce:
                            self.biclusteringDataSummary.ColumnChange( ci, self.GetNbrClusters()-1, self.biclusteringDataSummary.XT[i], self.bTranspose, False, False )
                        else:
                            self.biclusteringDataSummary.ColumnChange( ci, self.GetNbrClusters()-1, self.biclusteringDataSummary.XT[i], self.bTranspose, False, True )
                            bChangedOnce = True
                else:
                    c_i = self.U[i]
                    old_ci = oldU[i]
                    if c_i != cj:
                        if bChangedOnce:
                            self.biclusteringDataSummary.ColumnChange( ci, self.GetNbrClusters()-1, self.biclusteringDataSummary.X[i], self.bTranspose, False, False )
                        else:
                            self.biclusteringDataSummary.ColumnChange( ci, self.GetNbrClusters()-1, self.biclusteringDataSummary.X[i], self.bTranspose, False, True )
                            bChangedOnce = True
            #if self.bTranspose:
            #    print ci, self.L, self.colCnts
            #else:
            #    print ci, self.K, self.rowCnts

        #self.Verify("POST SPLIT")
        #print "Done Split",self.bTranspose

    def GetSplitCountsAndIndicators(self, ci):
        if self.bTranspose:
            I = self.colV[ci][:]
            oldU = self.V.copy()
            cnts = self.colCnts
        else:
            I = self.rowU[ci][:]
            oldU = self.U.copy()
            cnts = self.rowCnts
        return cnts,oldU,I

    def GetMergeCountsAndIndicators( self, ci, cj ):
        if self.bTranspose:
            I = self.colV[ci][:]
            I.extend(self.colV[cj][:])
            oldU = self.V.copy()
            cnts = self.colCnts
        else:
            I = self.rowU[ci][:]
            I.extend(self.rowU[cj][:])
            oldU = self.U.copy()
            cnts = self.rowCnts
        return cnts, oldU,I

    def MergeMove( self, ci, cj, N ):
        # propose merging ci and cj into ci
        self.ResetSplitMergeRestrictedLogProb()
        if cj < ci: # cj always greater than ci
            tmp = ci
            ci  = cj
            cj  = tmp
        cnts,oldU,I = self.GetMergeCountsAndIndicators( ci,cj )

        self.BackupClusters( ci, cj )
        
        """ compute reverse probabilities """
        self.RandomlyAssign( ci, cj, I )
        for t in range(self.nRestricted):
            self.UpdateSpecificParameters( ci, cj )
            bDivergedSplit,ccc = self.RestrictedGibbsSide( I, ci, cj )
            #if bDivergedSplit:
            #    print "DIVERGED: ",self.GetCount(ci), self.GetCount(cj)
        logaccb = GETLN(self.GetConc()) - GETGAMMALN(cnts[ci]) - GETGAMMALN(cnts[cj]) \
                  - self.RestrictedLogProbParamsPost \
                  + self.RestrictedLogProbParamsPrior \
                  + self.RestrictedLogProbData

        print "B", -GETGAMMALN(cnts[ci]) - GETGAMMALN(cnts[cj]),self.RestrictedLogProbData 
        """ compute forward probabilities """
        self.ResetSplitMergeRestrictedLogProb()
        self.MergeClusters( ci, cj )
        for t in range(1):
            self.UpdateSpecificParameters( ci )
            bDiverged = self.RestrictedGibbsSide( I, ci )

        print "*****************"
        logaccf = - GETGAMMALN(cnts[ci]) \
                  - self.RestrictedLogProbParamsPost \
                  + self.RestrictedLogProbParamsPrior \
                  + self.RestrictedLogProbData

        print "F", -GETGAMMALN(cnts[ci]) ,self.RestrictedLogProbData 
        print "MERGE F: ",logaccf," B: ",logaccb, " dif: ",logaccf-logaccb,bDivergedSplit
        #pdb.set_trace()
        if random.rand() > min( exp( logaccf-logaccb ), 1.0 ) or bDivergedSplit == True:
            self.RevertFromMerge( ci, cj )
            if self.bTranspose:
                self.biclusteringDataSummary.ColumnPutToEnd( cj, self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM )
            else:
                self.biclusteringDataSummary.ColumnPutToEnd( cj, self.biclusteringDataSummary.colDataSummaryMatrix.DSUM )
        else:
            print "--- MERGE ACCEPTED ---"
            if self.bTranspose:
                self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM = self.biclusteringDataSummary.ColumnMerge( ci,cj, self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM )
            else:
                self.biclusteringDataSummary.colDataSummaryMatrix.DSUM = self.biclusteringDataSummary.ColumnMerge( ci,cj, self.biclusteringDataSummary.colDataSummaryMatrix.DSUM )
        #self.Verify("END MERGE")

    def BackupClusters( self, ci, cj = None ):
        if cj is not None:
            self.bckdataSummary = self.bicDataSummary[[ci,cj],:].copy()
            self.bckParams      = self.bicParameters[[ci,cj],:].copy()
        else:
            self.bckdataSummary = self.bicDataSummary[ci,:].copy()
            self.bckParams      = self.bicParameters[ci,:].copy()
        if self.bTranspose:
            self.bck_V = self.V.copy()
            # back-up column
            self.bck_colV = []
            self.bck_colV.append( self.colV[ci][:] )
            if cj is not None:
                self.bck_colV.append( self.colV[cj][:] )
        else:
            # back-up row            
            self.bck_U    = self.U.copy()
            self.bck_rowU = []
            self.bck_rowU.append( self.rowU[ci][:] )
            if cj is not None:
                self.bck_rowU.append( self.rowU[cj][:] )

    def RandomlySplit( self, ci ):
        self.GenerateNewClusters()
        if self.bTranspose:
            K = self.L
            U = self.colV[ci][:]
            #print ci, K, self.colCnts
        else:
            U = self.rowU[ci][:]
            K = self.K
            #print ci, K, self.rowCnts
        assert self.GetCount(ci) > 1, "need at least 2 to split"
        if self.bTranspose:
            data_summary_list  = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[U[0]]
        else:
            data_summary_list  = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[U[0]]
        self.RemoveFromClusters( U[0], ci, data_summary_list ) 
        self.AddToClusters( U[0], K, data_summary_list )
        for i in U[1:]:
            if self.bTranspose:
                data_summary_list  = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[i]
            else:
                data_summary_list  = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[i]
            if random.rand() < 0.5:
                if self.GetCount(ci) > 1:
                    self.RemoveFromClusters( i, ci, data_summary_list ) 
                    self.AddToClusters( i, K, data_summary_list )
        #if self.bTranspose:
        #    print ci, self.L, self.colCnts
        #else:
        #    print ci, self.K, self.rowCnts

    def RandomlyAssign( self, ci, cj, U ):
        self.GenerateNewClusters()
        if self.bTranspose:
            K = self.L
        else:
            K = self.K
        for i in U:
            if self.bTranspose:
                data_summary_list  = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[i]
            else:
                data_summary_list  = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[i]
            curk = self.GetClusterId( i )
            if self.GetCount( curk ) > 1:
                if random.rand() < 0.5  and curk != ci:
                    self.RemoveFromClusters( i, curk, data_summary_list )
                    self.AddToClusters( i, ci, data_summary_list )
                else:
                    if curk != cj:
                        self.RemoveFromClusters( i, curk, data_summary_list )
                        self.AddToClusters( i, cj, data_summary_list )
        #print self.colCnts

    def MergeClusters( self, ci, cj ):
        if self.bTranspose:
            K = self.L
            U = array( self.colV[cj] )
        else:
            U = array( self.rowU[cj] )
            K = self.K
        for i in U:
            if self.bTranspose:
                data_summary_list  = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[i]
            else:
                data_summary_list  = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[i]
            self.RemoveFromClusters( i, cj, data_summary_list )
            self.AddToClusters( i, ci, data_summary_list ) 


    def RevertFromSplit( self, ci ):
        self.bicDataSummary[ci,:] = self.bckdataSummary.copy()
        self.bicParameters[ci,:]  = self.bckParams.copy()
        self.bicDataSummary       = self.bicDataSummary[:-1,:]
        self.bicParameters        = self.bicParameters[:-1,:]
        if self.bTranspose:
            self.V[self.bck_colV[0]] = ci
            V = self.bck_colV[0][:]
            newcol = []
            assert len(self.colV) == len(self.colCnts), "should be same"
            #print self.colV,self.colCnts
        
            self.colV.pop()
            self.colCnts[ci] += self.colCnts[-1]
            self.colCnts[-1] -= self.colCnts[-1]
            assert self.colCnts[-1] == 0, 'asd'
            self.L-=1
            self.colCnts.pop()
            self.colV[ci] = V
        else:
            self.U[self.bck_rowU[0]] = ci
            U = self.bck_rowU[0][:]
            self.rowU[ci] = U
            self.rowU.pop()
            self.rowCnts[ci] += self.rowCnts[-1]
            self.rowCnts[-1] -= self.rowCnts[-1]
            assert self.rowCnts[-1] == 0, 'asd'
            self.rowCnts.pop()
            self.K-=1

    def RevertFromMerge( self, ci, cj ):
        self.bicDataSummary       = concatenate( (self.bicDataSummary, zeros((1,self.bicDataSummary.shape[1]))) )
        self.bicParameters        = concatenate( (self.bicParameters, zeros((1,self.bicParameters.shape[1]))))
        self.bicDataSummary[[ci,-1],:] = self.bckdataSummary.copy()
        self.bicParameters[[ci,-1],:]  = self.bckParams.copy()
        if self.bTranspose:
            Ichange = find( self.bck_V == cj )
            self.V[Ichange] = self.L
            self.L+=1
            V1 = self.bck_colV[0][:]
            V2 = self.bck_colV[1][:]
            self.colCnts[ci] -= len(Ichange)
            self.colCnts.append(len(Ichange))
            self.colV.append(V2)
            self.colV[ci] = V1
        else:
            Ichange = find( self.bck_U == cj )
            self.U[Ichange] = self.K
            U1 = self.bck_rowU[0][:]
            U2 = self.bck_rowU[1][:]
            self.K+=1
            self.rowCnts[ci] -= len(Ichange)
            self.rowCnts.append(len(Ichange))
            self.rowU.append(U2)
            self.rowU[ci] = U1

    def RestrictedGibbsSide( self, I, ci, cj = None ):
        self.RestrictedLogProbGibbs       = 0.0
        self.RestrictedLogProbData        = 0.0
        N = len( I )
        P = random.permutation( N )
        for i in P:        
            if self.bTranspose:
                data_summary_list  = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[I[i]]
            else:
                data_summary_list  = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[I[i]]
            bDiverged, usable_c = self.RestrictedGibbsUpdate( I[i], ci, cj, data_summary_list )
            
            if bDiverged:
                return bDiverged, usable_c
        return False, ci

    def RestrictedGibbsUpdate( self, i, ci, cj, data_summary_list ):
        bDiverged = False
        # remove from current cluster
        if cj is None:
            
            self.RestrictedLogProbData  += self.GetLogProbCluster( ci, data_summary_list )
            #print  "RestrictedGibbsUpdate: ", self.RestrictedLogProbData, self.GetLogProbCluster( ci, data_summary_list )
            return bDiverged, ci
        else:
            assert ci < cj, 'order incorrect'
        oldk   = self.GetClusterId( i )
        curcnt = self.GetCount( oldk )
        if curcnt > 1:
            self.RemoveFromClusters( i, oldk, data_summary_list )
        else:
            self.RestrictedLogProbData  += self.GetLogProbCluster( oldk, data_summary_list )
            #print "returning cause curcnt = 1"
            return True, ci

        if cj == self.GetNbrClusters():
            self.RestrictedLogProbData  += self.GetLogProbCluster( ci, data_summary_list )
            print "RESTRICTED DIVERGED"
            if oldk == cj:
                # only ci is usable
                return True, ci
            else:
                # only cj is usable, but it is now cj - 1
                return True, cj-1
        # gather log probs of prior and likelihoods
        if self.bTranspose:
            logprior = log( array( [self.colCnts[ci], self.colCnts[cj]], dtype = float ))-log(self.colCnts[ci]+ self.colCnts[cj] )
        else:
            logprior = log( array( [self.rowCnts[ci], self.rowCnts[cj]], dtype = float ))-log(self.rowCnts[ci]+ self.rowCnts[cj] ) 

        logcomps  = array( [self.GetLogProbCluster( ci, data_summary_list ), self.GetLogProbCluster( cj, data_summary_list )], dtype = float )

        #print logprior, logcomps, logprior + logcomps
        # prob is proportional to prob under prior times data likelihood
        loglik = logprior + logcomps
        ls     = LogSum( loglik )
        lr     = loglik - ls
        #print "RES LR:", logprior, loglik, exp(lr), oldk,ci,cj ,logprior,logcomps
        if self.simspecs.bMaximize:
            newk   = argmax( lr )
        else:
            newk   = LogRnd( lr )
        #print "RES LR:", logprior, loglik, exp(lr), oldk,ci,cj ,logprior,logcomps,newk
        # gather stats
        self.RestrictedLogProbGibbs += lr[newk]
        self.RestrictedLogProbData  += logcomps[newk]
        #self.LogProbData  += ls

        # insert into new cluster
        if newk == 0:
            self.AddToClusters( i, ci, data_summary_list )
        else:
            self.AddToClusters( i, cj, data_summary_list )
        #print "CNTS REST.: ",self.rowCnts
        return bDiverged, ci

class BiclusteringGaussianModel( BiclusteringModel ):
    def __init__(self, clusteringPrior, modelPrior, data = None, names = None ):
        super( BiclusteringGaussianModel, self).__init__( clusteringPrior, modelPrior, data, names )
        self.bic_dof     = []
        self.bic_expprec = []
        self.row_dof     = []
        self.row_expprec = []
        self.col_dof     = []
        self.col_expprec = [] 

    def InitDataSummary( self ):
        return CreateGaussianDataSummary()

    def InitHyperParameters(self):
        super( BiclusteringGaussianModel, self).InitHyperParameters()
        """
           UPDATE BICLUSTER PARAMETERS
        """
        if self.mPrior.bicluster_params is not None:
            self.mPrior.bicluster_params.nu = self.mPrior.bic_hyper.dof_a / self.mPrior.bic_hyper.dof_b #random.gamma(self.mPrior.bic_hyper.dof_a, 1.0 / self.mPrior.bic_hyper.dof_b )
            assert self.mPrior.bicluster_params.nu > 0.0, "nu not positive" + str(self.mPrior.bicluster_params.nu)
            self.mPrior.bicluster_params.c  = self.mPrior.bic_hyper.expprec_a / self.mPrior.bic_hyper.expprec_b #random.gamma(self.mPrior.bic_hyper.expprec_a, 1.0 / self.mPrior.bic_hyper.expprec_b )
            assert self.mPrior.bicluster_params.c > 0.0, "c not positive" + str(self.mPrior.bicluster_params.c)
            self.bic_dof.append(self.mPrior.bicluster_params.nu)
            self.bic_expprec.append(self.mPrior.bicluster_params.c)
            
        """
           UPDATE ROW CLUSTER  PARAMETERS
        """
        if self.mPrior.rowcluster_params is not None:
            self.mPrior.rowcluster_params.nu = self.mPrior.row_hyper.dof_a/self.mPrior.row_hyper.dof_b #random.gamma(self.mPrior.row_hyper.dof_a, 1.0 / self.mPrior.row_hyper.dof_b )
            assert self.mPrior.rowcluster_params.nu > 0.0, "nu not positive" + str(self.mPrior.rowcluster_params.nu)
            self.mPrior.rowcluster_params.c  = self.mPrior.row_hyper.expprec_a / self.mPrior.row_hyper.expprec_b #random.gamma(self.mPrior.row_hyper.expprec_a, 1.0 / self.mPrior.row_hyper.expprec_b )
            assert self.mPrior.rowcluster_params.c > 0.0, "nu not positive" + str(self.mPrior.rowcluster_params.c)
            self.row_dof.append(self.mPrior.rowcluster_params.nu)
            self.row_expprec.append(self.mPrior.rowcluster_params.c)

        """
           UPDATE COL CLUSTER PARAMETERS
        """
        if self.mPrior.colcluster_params is not None:
            self.mPrior.colcluster_params.nu = self.mPrior.col_hyper.dof_a/self.mPrior.col_hyper.dof_b #random.gamma(self.mPrior.col_hyper.dof_a, 1.0 / self.mPrior.col_hyper.dof_b )
            assert self.mPrior.colcluster_params.nu > 0.0, 'gone bad'
            self.mPrior.colcluster_params.c  = self.mPrior.col_hyper.expprec_a/self.mPrior.col_hyper.expprec_b #random.gamma(self.mPrior.col_hyper.expprec_a, 1.0 / self.mPrior.col_hyper.expprec_b )
            assert self.mPrior.colcluster_params.c > 0.0, "nu not positive" + str(self.mPrior.colcluster_params.c)
            self.col_dof.append(self.mPrior.colcluster_params.nu)
            self.col_expprec.append(self.mPrior.colcluster_params.c)

    def Transpose( self ):
        super( BiclusteringGaussianModel, self).Transpose()
        if self.bTranspose:
            """ work on COLUMNS """
            pass
        else:
            """ work on ROWS """
            pass

    def UpdateHyperParameters(self):
        super( BiclusteringGaussianModel, self).UpdateHyperParameters()
        #return
        """
           UPDATE BICLUSTER PARAMETERS
        """
        if self.mPrior.bicluster_params is not None:
            prec = []
            for dparams in self.bicParameters:
                pvec = dparams[arange(2,len(dparams),3)]
                for p in pvec:
                    prec.append( p )
            #print "PREC",prec
            #figure(101)
            #clf()
            #hist( prec, 30 )
            #return
            self.UpdateExpectedPrec( self.mPrior.bic_hyper, self.mPrior.bicluster_params, prec )
            self.UpdateDof( self.mPrior.bic_hyper, self.mPrior.bicluster_params, prec )
            self.bic_dof.append(self.mPrior.bicluster_params.nu)
            #self.mPrior.bicluster_params.c = mean(prec)
            self.bic_expprec.append(self.mPrior.bicluster_params.c)

    def UpdateDof( self, hyper, params, prec ):
        #mlnu = sqrt( var(prec)*params.nu /2.0 )
        mlnu = 2.0*(params.c)/var(prec)
        params.nu, logprob = SliceDoF( params, hyper, prec )
        print "NU: ",params.nu," ML: ",mlnu
        #print "settting to EXPECTED"
        #params.nu = mlnu

    def UpdateExpectedPrec( self, hyper, params, prec ):
        mlc = (sum( prec ) + hyper.expprec_a/hyper.expprec_b)/float(len(prec)+1)
        params.c, logprob = SliceExpectedPrec( params, hyper, prec )
        print "EXPECTED PREC: ",params.c," ML: ",mlc
        #print "settting to EXPECTED"
        #params.c = mlc

    def ShowHyperparameters( self, fig ):
        super( BiclusteringGaussianModel, self).ShowHyperparameters(fig)
        figure(fig+1,figsize = (3,3))
        clf() 
        ioff()
        if len( self.bic_dof ) > 0:
            plot( self.bic_dof, 'b' )
            plot( self.bic_expprec, 'b--' )
            legend( ['nu','c'])
        if len( self.row_dof ) > 0:
            plot( self.row_dof, 'g' )
            plot( self.row_expprec, 'g--' )
        if len( self.col_dof ) > 0:
            plot( self.col_dof, 'r' )
            plot( self.col_expprec, 'r--' )
        ion() 
        draw()

    def GetPrecPosteriorError( self, other = [], otherother = [] ):
        bLongWay = False
        if len( other ) > 1:
            bLongWay = True
        errs = 0.0
        
        if bLongWay:
            assert len( otherother ) == len(other), "should be equal"
            for o,oo in zip( other, otherother ):
                mu = self.GetMean([o], [oo])
                errs += ( o.dataSummary.xsumsqrd + o.ndata*mu*mu-2.0*mu*o.dataSummary.xsum )
        else:
            mu   = self.GetMean( other, otherother )
            errs = self.dataSummary[2] + self.dataSummary[0]*mu*mu-2.0*mu*self.dataSummary[1]
        return errs

    def GetPrecPosteriorCounts( self, other = [], otherother = [] ):
        bLongWay = False
        if len( other ) > 1:
            bLongWay = True
        cnts = 0
        thisprec = self.prec
        #Log( "HACK" )
        if bLongWay:
            assert len( otherother ) == len(other), "should be equal"
            for o,oo in zip( other, otherother ):
                cnts     += o.dataSummary[0] * thisprec / float( o.GetPrec()+oo.GetPrec()+thisprec)
        else:
            prec = self.GetPrec( other, otherother )
            cnts = self.dataSummary[0]*thisprec / float( prec )
        return cnts

    def GetWeightedSumDeltaOther( self, other = [], otherother = [] ):
        bLongWay = False
        if len( other ) > 1:
            bLongWay = True
        deltasum = 0.0
        prec     = 0.0 
        if bLongWay:
            assert len( otherother ) == len(other), "should be equal"
            for o,oo in zip( other, otherother ):
                prec     += o.ndata*(o.GetPrec()+oo.GetPrec()+self.GetPrec())
                deltasum += (o.GetPrec()+oo.GetPrec()+self.GetPrec())*( o.datasum - (o.GetMean()+oo.GetMean())*o.ndata )
        else:
            mu   = self.GetMeanOther( other, otherother )
            prec = self.GetPrec( other, otherother )
            deltasum = prec*( self.dataSummary[1] - self.dataSummary[0]*mu )
            prec *= self.dataSummary[0]
        return deltasum, prec

    def GetPrec( self, other = [], otherother = [] ):
        prec = 0.0
        if self.bUse:
            prec += self.prec
        for o in other:  
            prec += o.GetPrec()
        for o in otherother:  
            prec += o.GetPrec()
        return prec

    def GetPrecOther( self, other = [], otherother = [] ):
        prec = 0.0
        for o in other:  
            prec += o.GetPrec()
        for o in otherother:  
            prec += o.GetPrec()
        return prec



    def PrecisionPrior( self ):
        if not self.bUse:
            return 0,0
        nu             = self.prior.nu
        expPrec        = self.prior.c

        shapeParam     = 0.5*nu 
        iscaleParam    = 0.5*nu/expPrec

        self.prec = random.gamma( shapeParam, 1.0/iscaleParam )
        if self.prec == 0.0:
            self.prec = shapeParam /iscaleParam 

    def PrecisionPosterior( self, other = [], otherother = [] ):
        #return 0.0,0.0
        if not self.bUse:
            return 0,0
        mu0            = self.prior.mu
        mu             = self.mu
        n              = self.dataSummary[0]#n
        nu             = self.prior.nu
        assert nu > 0.0, "nu is not positive"
        expPrec        = self.prior.c
        assert expPrec > 0.0, "expPrec is not positive"
        dPrecConst     = self.prior.a
        mPrecConst     = self.prior.b
        a     = self.prior.a
        b     = self.prior.b
        c     = self.prior.c
        sumx           = self.dataSummary[1]#xsum
        sumxsqred      = self.dataSummary[2]#xsumsqrd

        prec_cnts    = self.GetPrecPosteriorCounts( other, otherother )
        prec_error   = self.GetPrecPosteriorError(  other, otherother )
        #prec_error   = self.datasqsm - 2.0*self.datasum*self.mu + self.ndata*self.mu**2
        shapeParam   = 0.5*( float(prec_cnts+1) + nu )
        iscaleParam  = 0.5*nu/expPrec + 0.5*mPrecConst*(mu-mu0)**2 + 0.5*dPrecConst*prec_error
        #print "A",prec_error,prec_cnts,shapeParam,iscaleParam
        #iscaleParam    = 0.5*nu/expPrec + 0.5*mPrecConst*(mu-mu0)**2 + 0.5*dPrecConst*(sumxsqred+n*mu*mu-2.0*mu*sumx)
        #prec_error   = self.datasqsm - 2.0*self.datasum*self.mu + self.ndata*self.mu**2
        #shapeParam = 0.5*(self.ndata + 1.0 + nu )
        #iscaleParam = 0.5*(a*prec_error + b*(mu-mu0)**2 + nu/c )
        #print "B",prec_error,prec_cnts,shapeParam,iscaleParam
        assert iscaleParam > 0.0, "param should be positive" + str( iscaleParam )
        self.prec        = random.gamma( shapeParam, 1.0/iscaleParam )
        if self.prec == 0.0:
            self.prec = shapeParam /iscaleParam 
        #assert self.prec > 0.0, 'prec bad' + str(self.prec)
        #self.prec = 1.0
        #self.prec        = float( self.ndata ) / (self.datasqsm - 2*self.datasum*self.mu + self.ndata*self.mu**2)
        logprobprior     = logprobgamma(self.prec, nu/2.0, expPrec )
        logprobposterior = logprobgamma(self.prec, shapeParam, iscaleParam )
        #print "PREC: ",self.prec, nu/2.0, expPrec , shapeParam, iscaleParam, logprobprior, logprobposterior
        #print "PREC: ", self.prec, shapeParam/iscaleParam
        return logprobprior, logprobposterior

    def MeanPrior( self ):
        if not self.bUse:
            return 0,0
        mu0            = self.prior.mu
        s              = self.prec
        b              = self.prior.b

        expectedMuWeight = b*s
        expectedMu       = b*s*mu0/expectedMuWeight
        self.mu          = random.normal( loc=expectedMu, scale=1.0/sqrt(expectedMuWeight) )

    def MeanPosterior( self, other = [], otherother = [] ):
        if not self.bUse:
            return 0,0
        mu0            = self.prior.mu
        s              = self.prec
        n              = self.dataSummary[0]
        a  = self.prior.a
        b = self.prior.b
        sumx           = self.dataSummary[1]

        muother, prec = self.GetWeightedSumDeltaOther( other, otherother )
        #prec    = self.GetPrec(  other, otherother )

        expectedMuWeight = a*prec + b*s
        expectedMu       = (a*muother + b*s*mu0)/expectedMuWeight

        self.mu          = random.normal( loc=expectedMu, scale=1.0/sqrt(expectedMuWeight) )
        logprobprior     = logprobnormal(self.mu, mu0, 1/(s*b) )
        logprobposterior = logprobnormal(self.mu, expectedMu, 1.0/expectedMuWeight )
        return logprobprior, logprobposterior

    def Verify(self, MSG = "" ):
        dlen = GaussianDataSummaryLength()
        N = self.data.nEntries - self.data.nMissing
        #print self.U, self.V
        for JJ in self.rowU:
            #print JJ
            assert len(JJ) > 0, 'prob'+str(JJ)
        for JJ in self.colV:
            #print JJ
            assert len(JJ) > 0, 'prob'+str(JJ)
        #Log( "VERIFY -- U/V: " + str( unique(self.U) ) + " " + str( unique(self.V) ) + " (" + MSG + ")")
        #Log( "VERIFY -- K/L: " + str( self.K ) + " " + str( self.L ) + " (" + MSG + ")")
        #Log( "VERIFY -- CNT: " + str( self.rowCnts ) + ", " + str( sum(self.rowCnts) ) +" "+str( self.colCnts )+ ", " + str( sum(self.colCnts) ) + " (" + MSG + ")")
        sumx = 0.0
        sumsqx = 0.0
        n = 0
        nvec = self.bicDataSummary[:,arange(0,self.bicDataSummary.shape[1],3)]
        assert all(nvec>=0),'Some counts gone ngatve' + str(nvec)
        n = self.bicDataSummary.sum(0)[arange(0,self.bicDataSummary.shape[1],3)].sum()
        bError = False
        K = self.K
        #print self.biclusteringDataSummary.colDataSummaryMatrix.DSUM.sum(0)
        #print self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM.sum(0)
        B = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[0,arange(0,self.biclusteringDataSummary.colDataSummaryMatrix.DSUM.shape[1],3)]
        assert all(B>=0),'B bad' + str(B)
        B = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[0,arange(0,self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM.shape[1],3)]
        assert all(B>=0),'B bad' + str(B)
        if self.bTranspose:
            assert self.bicDataSummary.shape[0] == self.L, 'ww'
            assert self.bicDataSummary.shape[1] == self.K*3, 'ww'
            for rowdsum, k in zip( self.bicDataSummary, range(self.L) ):
                nk  = rowdsum[arange(0,self.bicDataSummary.shape[1],3)].sum()
                nvec = rowdsum[arange(0,self.bicDataSummary.shape[1],3)]
                assert all(nvec>=0),'Some counts gone ngatve' + str(nvec)
                nkcheck  = self.biclusteringDataSummary.colDataSummaryMatrix.DSUM[self.colV[k],:].sum(0)[arange(0,self.bicDataSummary.shape[1],3)]
                if nk != nkcheck.sum():
                    bError=True
                    print '   problem with row cluster ',k,nk,nkcheck,nvec
                else:
                    pass #print 'NO problem with row cluster ',k,nk,nkcheck,nvec
        else:
            assert self.bicDataSummary.shape[0] == self.K, 'ww'
            assert self.bicDataSummary.shape[1] == self.L*3, 'ww'
            for rowdsum, k in zip( self.bicDataSummary, range(self.K) ):
                nk  = rowdsum[arange(0,self.bicDataSummary.shape[1],3)].sum()
                nkcheck  = self.biclusteringDataSummary.rowDataSummaryMatrix.DSUM[self.rowU[k],:].sum(0)[arange(0,self.bicDataSummary.shape[1],3)]
                #assert nk == nkcheck, 'should be same ' + str(nk) + " " + str(nkcheck)
                nvec = rowdsum[arange(0,self.bicDataSummary.shape[1],3)]
                assert all(nvec>=0),'Some counts gone ngatve' + str(nvec)
                if nk != nkcheck.sum():
                    bError=True
                    print '   problem with row cluster ',k,nk,nkcheck.sum(),nkcheck,nvec
                else:
                    pass #print 'NO problem with row cluster ',k,nk,nkcheck.sum(),nkcheck,nvec
        
        #for r, U in zip( self.Biclusters, self.rowU ):
        #    for bicluster,V in zip( r, self.colV):
        #        n      += bicluster.dataSummary[0]
        #        sumx   += bicluster.dataSummary[1] 
        #        sumsqx += bicluster.dataSummary[2]
        #        #print bicluster.dataSummary, len(U), len(V), len(U)*len(V)
        assert N == n, 'should be the same: '+ str(N) + "," + str(n)
        assert bError is False,"problem with clusters"
        #Log( "VERIFY")
     
"""
   CONJUGATE GAUSSIAN MODEL
"""   
class BiclusteringConjugateGaussianModel( BiclusteringGaussianModel ):
    def __init__(self, clusteringPrior, modelPrior, data = None, names = None  ):
        super( BiclusteringConjugateGaussianModel, self).__init__( clusteringPrior, modelPrior, data, names )
        self.conjmu      = []
        self.param_names = ['mu','conjmu','prec']
        self.nparams     = 3
        self.nsummary    = 3 # ndata, xsum, xsumsqrd

    def InitParams( self, rowsummary, rowsummarytot = None ):
        D = len(rowsummary)
        nClusters = D / self.nsummary

        a = self.mPrior.bicluster_params.a
        b = self.mPrior.bicluster_params.b
        c = self.mPrior.bicluster_params.c
        nu = self.mPrior.bicluster_params.nu
        mu = self.mPrior.bicluster_params.mu

        conjmu      = self.UpdateMean( a,b,mu, rowsummary[arange(0,D,self.nsummary)], rowsummary[arange(1,D,self.nsummary)] )
        shapeParam  = 0.5*nu*ones(nClusters,dtype=float) 
        iscaleParam = 0.5*nu*ones(nClusters,dtype=float) / c

        prec = self.UpdatePrec( nu,c )

        DD = self.nparams*nClusters
        rowparams = zeros( DD, dtype = float )
        rowparams[arange(0,DD,self.nparams)] = conjmu
        rowparams[arange(2,DD,self.nparams)] = conjmu
        rowparams[arange(3,DD,self.nparams)] = prec
        return rowparams, None

    def GenerateRandomBiclusterRow( self, params = None ):
        D = self.bicDataSummary.shape[1]
        rowsummary = zeros( D, dtype = float )
        return self.InitParams( rowsummary )[0]

    def Transpose( self ):
        super( BiclusteringConjugateGaussianModel, self).Transpose()
        #print 'pre-transpose'
        if self.bTranspose:
            """ work on COLUMNS """
            bicDataSummary = zeros( (self.L, self.K*self.nsummary), dtype=float )
            bicParameters  = zeros( (self.L, self.K*self.nparams), dtype=float )
            for k in range(self.K ):
                for l in range(self.L):
                    bicDataSummary[l,k*self.nsummary:k*self.nsummary+self.nsummary] = self.bicDataSummary[k,l*self.nsummary:l*self.nsummary+self.nsummary]
                    bicParameters[l,k*self.nparams:k*self.nparams+self.nparams] = self.bicParameters[k,l*self.nparams:l*self.nparams+self.nparams]
            #pass
        else:
            """ work on ROWS """
            bicDataSummary = zeros( (self.K, self.L*self.nsummary), dtype=float )
            bicParameters  = zeros( (self.K, self.L*self.nparams), dtype=float )
            for k in range(self.K ):
                for l in range(self.L):
                    bicDataSummary[k,l*self.nsummary:l*self.nsummary+self.nsummary] = self.bicDataSummary[l,k*self.nsummary:k*self.nsummary+self.nsummary]
                    bicParameters[k,l*self.nparams:l*self.nparams+self.nparams] = self.bicParameters[l,k*self.nparams:k*self.nparams+self.nparams]

        self.bicDataSummary = bicDataSummary
        #print 'post-transpose'
        self.bicParameters  = bicParameters

    def GetLogProbClusterVec(self, testdsum, rowparams, rowdsum = None ):
        """ compute logprob of y | self """
        #print "A"
        a = self.mPrior.bicluster_params.a
        b = self.mPrior.bicluster_params.b
        c = self.mPrior.bicluster_params.c
        nu = self.mPrior.bicluster_params.nu
        mu0 = self.mPrior.bicluster_params.mu
        EE = len(rowparams)
        mu = rowparams[arange(0,EE,self.nparams)]
        #logprob = zeros(1,dtype=float)
        DD        = len(testdsum)
        dim       = testdsum[arange(0,DD,self.nsummary)] 
        xsum      = testdsum[arange(1,DD,self.nsummary)]
        xsumsqrd  = testdsum[arange(2,DD,self.nsummary)]
        if rowdsum is None:
            ndim      = 0.0 #zeros( len(dim), dtype=float )
        else:
            assert DD == len(rowdsum), 'same size!'
            ndim      = rowdsum[arange(0,DD,self.nsummary)]

        if rowdsum is None:
            A       = a*c;
            B       = A*a / (a*(ndim+dim)+b)
            F       = a / (a*(ndim+dim)+b)
            MAL = A*(xsumsqrd - 2*xsum*mu + dim*mu*mu) - B*(xsum-dim*mu)**2
            lp1  = GETGAMMALN(  0.5*(nu + dim)) - GETGAMMALN( 0.5*nu ) #self.ComputeGammaln(nu,dim) #special.gammaln( (nu + dim)/2.0 ) - special.gammaln( nu / 2.0 ) 
            lp2 = lp1
            lp1 += 0.5*( (dim-1.0)*log(A ) +GETLN( A-dim*B ) - dim*GETLN(nu*pi) - (nu+dim)*GETLN(1.0+MAL/nu) ) #self.ComputeConstants(A,B,ndim,dim,a,b,nu,MAL) #0.5*( dim*log(A ) +log( (ndim*a+b)/(a*(dim+ndim) + b) ) - dim*log(nu*pi) - (nu+dim)*log(1.0+MAL/nu) )

        if rowdsum is not None:
            s = c
            n1 = dim
            sqx1 = xsumsqrd
            sx1 = xsum
            n2 = ndim
            sqx2 = rowdsum[arange(2,DD,self.nsummary)]
            sx2  = rowdsum[arange(1,DD,self.nsummary)]
            mn = ((a*sx1+b*mu0)/(a*n2+b))
            bb = a/((n1+n2)*a+b)
            """err = GETGAMMALN(n1) - GETGAMMALN(nu) -0.5*n1*log(pi*nu) + 0.5*n1*log(s) + 0.5*n1*log(a) + 0.5*log(b+a*n2) -0.5*log((n1+n2)*a+b)
            err = err - 0.5*(n1+nu)*log(1+(1/nu)*s*a*(sqx1-sx1*sx1*bb+mn*mn*n1-bb*mn*mn*n1*n1-2*sx1*mn+2*bb*sx1*mn*n1))"""
            err =  GETGAMMALN(n1)
            err = err - GETGAMMALN(nu)
            err = err -0.5*n1*log(pi*nu)
            err = err + 0.5*n1*log(s*a)
            err = err + 0.5*log(b+a*n2)
            err = err -0.5*log((n1+n2)*a+b)
            err = err - 0.5*(n1+nu)*log(1+(1/nu)*s*a*(sqx1-bb*(sx1*sx1+mn*mn*n1*n1-2*sx1*mn*n1)+mn*mn*n1-2*sx1*mn))
            #print err, lp1
            return sum(err)
        return sum(lp1)

    def UpdateMean( self, a,b,mu,dim,xsum ):
        return  (a*xsum + b*mu ) / (a*dim + b)

    def UpdatePrec( self, nu,c, params = None, dsum = None ):
        shapeParam  = 0.5*nu
        iscaleParam = 0.5*nu / c
        if dsum is not None:
            a     = self.mPrior.bicluster_params.a
            b     = self.mPrior.bicluster_params.b
            DD    = len(dsum)
            prec_cnts = dsum[arange(0,DD,self.nsummary)] 
            xsum      = dsum[arange(1,DD,self.nsummary)]
            xsumsqrd  = dsum[arange(2,DD,self.nsummary)]
            mu        = params[arange(2,len(params),self.nparams)]
            mu0       = self.mPrior.bicluster_params.mu
            #prec_cnts    = self.GetPrecPosteriorCounts( other, otherother )
            #prec_error   = self.GetPrecPosteriorError(  other, otherother )
            #self.dataSummary[2] + self.dataSummary[0]*mu*mu-2.0*mu*self.dataSummary[1]
            prec_error   = xsumsqrd - 2.0*xsum*mu + prec_cnts*mu*mu
            shapeParam   += 0.5*( prec_cnts+1)
            iscaleParam  +=  0.5*b*(mu-mu0)**2 + 0.5*a*prec_error
        return random.gamma( shapeParam, 1.0/iscaleParam )

    def UpdateParamsVector( self, rowsummary, rowparams, specificClusters = None  ):
        D = len(rowsummary)
        nClusters = D / self.nsummary

        a = self.mPrior.bicluster_params.a
        b = self.mPrior.bicluster_params.b
        c = self.mPrior.bicluster_params.c
        nu = self.mPrior.bicluster_params.nu
        mu = self.mPrior.bicluster_params.mu

        conjmu      = self.UpdateMean( a,b,mu, rowsummary[arange(0,D,self.nsummary)], rowsummary[arange(1,D,self.nsummary)] )
        shapeParam  = 0.5*nu*ones(nClusters,dtype=float) 
        iscaleParam = 0.5*nu*ones(nClusters,dtype=float) / c

        prec = self.UpdatePrec( nu,c, rowparams, rowsummary )

        DD = self.nparams*nClusters
        rowparams *= 0.0 #zeros( DD, dtype = float )
        rowparams[arange(0,DD,self.nparams)] = conjmu
        rowparams[arange(1,DD,self.nparams)] = conjmu
        rowparams[arange(2,DD,self.nparams)] = prec
        #return rowparams
        return 0.0, 0.0

    def UpdateParams( self ):
        assert False, 'not ready'
        self.conjmu = self.UpdateMean()
        #def UpdateParams( self, other = [], otherother = [] ):
        nRepeats = 1
        for t in range(nRepeats):
            #pdb.set_trace()
            logprobPriora, logprobPosteriora = self.PrecisionPosterior( other, otherother )
            logprobPriorb, logprobPosteriorb = self.MeanPosterior( other, otherother )
        logprobPrior     = logprobPriora + logprobPriorb
        logprobPosterior = logprobPosteriora + logprobPosteriorb

        self.mu = self.conjmu
        return 0.0, 0.0

    def SampleDatum( self, u, v ):
        assert self.bTranspose is False, 'only call of working on rows'
        return self.bicParameters[u][1+self.nparams*v]

    def GetMergeCandidates( self, ci ):
        # extract a matrix of bicluster means
        M = []
        for params in self.bicParameters:
           mns = params[arange(0,len(params),self.nparams)]
           M.append(mns)
        M = array(M)
        #print self.K,self.L,M.shape,M
        # compute correlation between mean vectors
        if M.shape[1] == 1:
            C = multiply(M,M.T)
        else:
            C = corrcoef(M)
            
        #print C
        C = C[ci]
        CC = C.copy()
        C -= min(C)
        C = C / C.sum()
        
        cj = ci
        nMax = 10
        n = 0
        while cj == ci and n < nMax:
            #print C, GETLN( C+ 0.00001 )
            cj  = LogRnd( GETLN( C + 0.00001 ) )
            n+=1
        if cj > ci: 
            tmp = ci
            cj = ci
            ci = tmp
        # sample candidates based on correlation
        #print "MERGING: ", ci,cj," because corr ",CC[cj],CC,GETLN( C + 0.00001 )
        return ci, cj
"""
   NON-CONJUGATE GAUSSIAN MODEL
"""
class BiclusteringNonConjugateGaussianModel( BiclusteringGaussianModel ):
    def __init__(self, clusteringPrior, modelPrior, data = None, names = None  ):
        super( BiclusteringNonConjugateGaussianModel, self).__init__( clusteringPrior, modelPrior, data, names )
        self.param_names = ['mu','prec']
        self.nparams     = len(self.param_names)
        self.nsummary    = 3 # ndata, xsum, xsumsqrd


    def InitParams( self, rowsummary, rowsummarytot = None ):
        D = len(rowsummary)
        nClusters = D / self.nsummary

        a   = self.mPrior.bicluster_params.a
        b   = self.mPrior.bicluster_params.b
        c   = self.mPrior.bicluster_params.c
        nu  = self.mPrior.bicluster_params.nu
        mu0 = self.mPrior.bicluster_params.mu


        prec = self.PosteriorPrec( nu,c )
        mu,lpprior,lppost      = self.PosteriorMean( a,b, c, mu0,prec, rowsummary )
        #conjmu      = self.UpdateMean( a,b,mu, rowsummary[arange(0,D,self.nsummary)], rowsummary[arange(1,D,self.nsummary)] )
        shapeParam  = 0.5*nu*ones(nClusters,dtype=float) 
        iscaleParam = 0.5*nu*ones(nClusters,dtype=float) / c


        DD = self.nparams*nClusters
        bicrowparams = zeros( DD, dtype = float )
        bicrowparams[arange(0,DD,self.nparams)] = mu
        bicrowparams[arange(1,DD,self.nparams)] = prec
        #bicrowparams[arange(3,DD,self.nparams)] = prec
        return bicrowparams,None

    def InitColParams( self, colsummarytot ):
        if self.mPrior.colcluster_params is None:
            return None
        assert len( colsummarytot ) == self.nsummary, 'should be total summary'

        a = self.mPrior.colcluster_params.a
        b = self.mPrior.colcluster_params.b
        c = self.mPrior.colcluster_params.c
        nu = self.mPrior.colcluster_params.nu
        mu = self.mPrior.colcluster_params.mu

        mu      = self.UpdateMean( a,b,mu, rowsummary[arange(0,D,self.nsummary)], rowsummary[arange(1,D,self.nsummary)] )

        """ sample precision from prior"""
        shapeParam  = 0.5*nu 
        iscaleParam = 0.5*nu / c
        prec = self.UpdatePrec( nu,c )

        colparams = zeros( self.nparams, dtype = float )
        colparams[0] = mu
        colparams[1] = prec
        return colparams

    def GenerateRandomBiclusterRow( self, params = None ):
        D = self.bicDataSummary.shape[1]
        rowsummary = zeros( D, dtype = float )
        return self.InitParams( rowsummary )[0]

    def Transpose( self ):
        super( BiclusteringNonConjugateGaussianModel, self).Transpose()
        #print 'pre-transpose',self.bicDataSummary
        if self.bTranspose:
            """ work on COLUMNS """
            bicDataSummary = zeros( (self.L, self.K*self.nsummary), dtype=float )
            bicParameters  = zeros( (self.L, self.K*self.nparams), dtype=float )
            for k in range(self.K ):
                for l in range(self.L):
                    bicDataSummary[l,k*self.nsummary:k*self.nsummary+self.nsummary] = self.bicDataSummary[k,l*self.nsummary:l*self.nsummary+self.nsummary]
                    bicParameters[l,k*self.nparams:k*self.nparams+self.nparams] = self.bicParameters[k,l*self.nparams:l*self.nparams+self.nparams]
            #pass
        else:
            """ work on ROWS """
            bicDataSummary = zeros( (self.K, self.L*self.nsummary), dtype=float )
            bicParameters  = zeros( (self.K, self.L*self.nparams), dtype=float )
            for k in range(self.K ):
                for l in range(self.L):
                    bicDataSummary[k,l*self.nsummary:l*self.nsummary+self.nsummary] = self.bicDataSummary[l,k*self.nsummary:k*self.nsummary+self.nsummary]
                    bicParameters[k,l*self.nparams:l*self.nparams+self.nparams] = self.bicParameters[l,k*self.nparams:k*self.nparams+self.nparams]

        self.bicDataSummary = bicDataSummary
        #print 'post-transpose',self.bicDataSummary
        self.bicParameters  = bicParameters

    def GetLogProbClusterVec(self, testdsum, rowparams, rowdsum = None ):
        """ compute logprob of y | self """
        #print "A"
        a = self.mPrior.bicluster_params.a
        b = self.mPrior.bicluster_params.b
        c = self.mPrior.bicluster_params.c
        nu = self.mPrior.bicluster_params.nu
        mu = self.mPrior.bicluster_params.mu
        EE = len(rowparams)
        mu   = rowparams[arange(0,EE,self.nparams)]
        prec = rowparams[arange(1,EE,self.nparams)]
        #logprob = zeros(1,dtype=float)
        DD        = len(testdsum)
        dim       = testdsum[arange(0,DD,self.nsummary)] 
        xsum      = testdsum[arange(1,DD,self.nsummary)]
        xsumsqrd  = testdsum[arange(2,DD,self.nsummary)]

        lp1 = -0.5*prec*(xsumsqrd - 2*xsum*mu + dim*mu*mu) #self.ComputeMAL(A, B, xsumsqrd, xsum, mu, dim) #A*(xsumsqrd - 2*xsum*mu + dim*mu*mu) - B*(xsum-dim*mu)**2
        lp1 += 0.5* dim*GETLN( prec ) - 0.5*dim*GETLN(2.0*pi) #+GETLN( (ndim*a+b)/(a*(dim+ndim) + b) ) - dim*GETLN(nu*pi) - (nu+dim)*GETLN(1.0+MAL/nu) ) #self.ComputeConstants(A,B,ndim,dim,a,b,nu,MAL) #0.5*( dim*log(A ) +log( (ndim*a+b)/(a*(dim+ndim) + b) ) - dim*log(nu*pi) - (nu+dim)*log(1.0+MAL/nu) )
        return sum(lp1)

    def UpdateHyperParameters(self):
        super( BiclusteringGaussianModel, self).UpdateHyperParameters()
        #return
        """
           UPDATE BICLUSTER PARAMETERS
        """
        if self.mPrior.bicluster_params is not None:
            prec = []
            for dparams in self.bicParameters:
                pvec = dparams[arange(1,len(dparams),2)]
                for p in pvec:
                    prec.append( p )
            #print "PREC",prec
            #figure(101)
            #clf()
            #hist( prec, 30 )
            #return
            self.UpdateExpectedPrec( self.mPrior.bic_hyper, self.mPrior.bicluster_params, prec )
            self.UpdateDof( self.mPrior.bic_hyper, self.mPrior.bicluster_params, prec )
            self.bic_dof.append(self.mPrior.bicluster_params.nu)
            #self.mPrior.bicluster_params.c = mean(prec)
            self.bic_expprec.append(self.mPrior.bicluster_params.c)

    def UpdateMean( self, a,b,mu,dim,xsum ):
        return  (a*xsum + b*mu ) / (a*dim + b)



    def UpdateParamsVector( self, rowsummary, rowparams, specificClusters = None  ):
        D = len(rowsummary)
        nClusters = D / self.nsummary

        DD = self.nparams*nClusters
        assert DD == len(rowparams)
        a = self.mPrior.bicluster_params.a
        b = self.mPrior.bicluster_params.b
        c = self.mPrior.bicluster_params.c
        nu = self.mPrior.bicluster_params.nu
        mu0 = self.mPrior.bicluster_params.mu

        mu,lpprior,lppost      = self.PosteriorMean( a,b, c, mu0,rowparams[arange(1,DD,self.nparams)], rowsummary )
        shapeParam  = 0.5*nu*ones(nClusters,dtype=float) 
        iscaleParam = 0.5*nu*ones(nClusters,dtype=float) / c

        prec = self.PosteriorPrec( nu, c, mu, rowsummary )

        rowparams *= 0.0 #zeros( DD, dtype = float )
        rowparams[arange(0,DD,self.nparams)] = mu
        rowparams[arange(1,DD,self.nparams)] = prec
        return 0.0, 0.0

    def PosteriorMean( self, a, b, c, mu0,prec, rowsummary ):

        DD = len(rowsummary)
        n         = rowsummary[arange(0,DD,self.nsummary)] 
        sumx      = rowsummary[arange(1,DD,self.nsummary)]
        xsumsqrd  = rowsummary[arange(2,DD,self.nsummary)]

        expectedMuWeight = a*prec*n + b*c
        expectedMu       = (a*sumx*prec + b*c*mu0)/expectedMuWeight

        mu          = random.normal( loc=expectedMu, scale=1.0/sqrt(expectedMuWeight) )
        logprobprior     = logprobnormal(mu, mu0, 1/(c*b) )
        logprobposterior = logprobnormal(mu, expectedMu, 1.0/expectedMuWeight )
        return mu,logprobprior, logprobposterior

    def PosteriorPrec( self, nu,c, mu=None, dsum = None ):
        shapeParam  = 0.5*nu
        iscaleParam = 0.5*nu / c
        if dsum is not None:
            a     = self.mPrior.bicluster_params.a
            b     = self.mPrior.bicluster_params.b
            DD    = len(dsum)
            prec_cnts = dsum[arange(0,DD,self.nsummary)] 
            xsum      = dsum[arange(1,DD,self.nsummary)]
            xsumsqrd  = dsum[arange(2,DD,self.nsummary)]
            #mu        = params[arange(0,len(params),self.nparams)]
            #prec      = params[arange(1,len(params),self.nparams)] 
            mu0       = self.mPrior.bicluster_params.mu
            #prec_cnts    = self.GetPrecPosteriorCounts( other, otherother )
            #prec_error   = self.GetPrecPosteriorError(  other, otherother )
            #self.dataSummary[2] + self.dataSummary[0]*mu*mu-2.0*mu*self.dataSummary[1]
            prec_error   = xsumsqrd - 2.0*xsum*mu + prec_cnts*mu*mu
            shapeParam   += 0.5*( prec_cnts+1)
            iscaleParam  +=  0.5*b*(mu-mu0)**2 + 0.5*a*prec_error
        return random.gamma( shapeParam, 1.0/iscaleParam )

    def UpdateParams( self ):
        assert False, 'not ready'
        self.UpdateMean()
        #def UpdateParams( self, other = [], otherother = [] ):
        nRepeats = 1
        for t in range(nRepeats):
            #pdb.set_trace()
            logprobPriora, logprobPosteriora = self.PrecisionPosterior( other, otherother )
            logprobPriorb, logprobPosteriorb = self.MeanPosterior( other, otherother )
        logprobPrior     = logprobPriora + logprobPriorb
        logprobPosterior = logprobPosteriora + logprobPosteriorb

        self.mu = self.conjmu
        return 0.0, 0.0

    def SampleDatum( self, u, v ):
        assert self.bTranspose is False, 'only call of working on rows'
        return self.bicParameters[u][0+self.nparams*v]

    def GetMergeCandidates( self, ci ):
        # extract a matrix of bicluster means
        M = []
        for params in self.bicParameters:
           mns = params[arange(0,len(params),self.nparams)]
           M.append(mns)
        M = array(M)
        #print self.K,self.L,M.shape,M
        # compute correlation between mean vectors
        if M.shape[1] == 1:
            C = multiply(M,M.T)
        else:
            C = corrcoef(M)
            
        #print C
        C = C[ci]
        CC = C.copy()
        C -= min(C)
        sss = C.sum()
        if sss > 0:
          C = C / sss
        #C = C / C.sum()
        
        cj = ci
        nMax = 10
        n = 0
        while cj == ci and n < nMax:
            #print C, GETLN( C+ 0.00001 )
            cj  = LogRnd( GETLN( C + 0.00001 ) )
            n+=1
        if cj > ci: 
            tmp = ci
            cj = ci
            ci = tmp
        # sample candidates based on correlation
        #print "MERGING: ", ci,cj," because corr ",CC[cj],CC,GETLN( C + 0.00001 )
        return ci, cj

    def GetMeans( self ):
        MNS = []
        for rowparams in self.bicParameters:
            MNS.append( rowparams[arange(0,len(rowparams),self.nparams)] )
        return array(MNS)
    def GetPrecisions( self ):
        PRS = []
        for rowparams in self.bicParameters:
            PRS.append( rowparams[arange(1,len(rowparams),self.nparams)] )
        return array(PRS)
    def GetSizes( self ):
        SZS = []
        for rowparams in self.bicDataSummary:
            SZS.append( rowparams[arange(0,len(rowparams),self.nsummary)] )
        return array(SZS,dtype=int)
