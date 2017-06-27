from bic_math import safecorrcoef
#from matplotlib.mlab import load
from pylab import hist, matshow, cm, hlines, legend,vlines, show, draw, axis, figure, ion, ioff, clf, hold, colorbar, prism,autumn, plot,imshow

from numpy import random, array, zeros, matrix, log, multiply, inf, ones, floor, tile
from scipy import sparse

import pdb

class dataname( object ):
    def __init__( self, rownames, altrownames, colnames, altcolnames, rowdesc = None ):
        self.rownames = rownames
        self.altrownames = altrownames 
        self.altcolnames = altcolnames 
        self.colnames = colnames
        self.rowdesc  = rowdesc
        self.descdict = {}
        if rowdesc is not None:
            assert len( rowdesc ) == len( rownames), 'should be same size'
            for rn,rd in zip( self.rownames, self.rowdesc ):
                self.descdict[ rn ] = rd

def LoadData( fname, percentTest = 0.0, maxdims = [-1,-1] ):
    """
       For now, just load a full matrix, later check and load sparse (missing entries).
    """
    import numpy
    try:
        X = numpy.load( fname + '.pkl' )
    except:
        X = load( fname + '.txt' )
    nRows,nCols = X.shape
    if maxdims[0] > 0:
        nRows = min( nRows, maxdims[0] )
    if maxdims[1] > 0:
        nCols = min( nCols, maxdims[1] )

    X = X[0:nRows,:]
    X = X[:,0:nCols]
    fid = open( fname + '.rownames')
    Y = fid.readlines()
    fid.close()

    rownames = []
    for rn in Y:
        rownames.append( rn.split( '\n' )[0] )


    fid = open( fname + '.colnames')
    Y = fid.readlines()
    fid.close()
    colnames = []
    for rn in Y:
        colnames.append( rn.split( '\n' )[0] )

    try:
        fid = open( fname + '.altrownames')
        Y = fid.readlines()
        fid.close()
        altrownames = []
        for rn in Y:
            altrownames.append( rn.split( '\n' )[0] )
    except:
        altrownames = rownames[:]
    try:
        fid = open( fname + '.altcolnames')
        Y = fid.readlines()
        fid.close()
        altcolnames = []
        for rn in Y:
            altcolnames.append( rn.split( '\n' )[0] )
    except:
        altcolnames = colnames[:]
    try:
        fid = open( fname + '.rowdesc')
        Y = fid.readlines()
        fid.close()
        rowdesc = []
        for rn in Y:
            rowdesc.append( rn.split( '\n' )[0] )
    except:
        rowdesc = rownames[:]

    rownames2 = []
    altrownames2 = []
    rowdesc2 = []
    for i in range( nRows ):
        rownames2.append( rownames[i] )
        altrownames2.append( altrownames[i] )
        rowdesc2.append( rowdesc[i] )
    rownames = rownames2
    altrownames = altrownames2
    rowdesc = rowdesc2
    colnames2 = []
    altcolnames2 = []
    for i in range( nCols ):
        colnames2.append( colnames[i] )
        altcolnames2.append( altcolnames[i] )
    colnames    = colnames2
    altcolnames = altcolnames2
        
    missingVal = 999.0
    return BiclusteringFullData( X, missing=missingVal, percentTest=percentTest ), dataname( rownames, altrownames, colnames, altcolnames, rowdesc )
        
def LoadSparse( fname, dtype ):
    fid = open( fname )
    allLines = fid.readlines()
    fid.close()
    topline = allLines[0].split()
    nnz = int( eval( allLines[1].split()[0] ) )
    nRows = int( eval(topline[0]) )
    nCols = int( eval(topline[1]) )
    X  = sparse.dok_matrix((nRows,nCols) )
    #XX = sparse.dok_matrix((nRows,nCols) )

    #pdb.set_trace()
    for i,L in zip(range(nnz), allLines[2:] ):
        z = L.split()
        X[int(z[0])-1,int(z[1])-1] = eval(z[2])
        #XX[int(z[0])-1,int(z[1])-1] =X[int(z[0])-1,int(z[1])-1]**2
        if i > 10000:  
            return array(X.todense(), dtype=int), X.getnnz() #,XX
    return X, X.getnnz()
    
class BiclusteringSparseData( object ):
    def __init__( self, X, nnz ):
        self.X   = X 
        self.XT  = X.T
        #self.XX  = X*X
        #self.XXT = self.XX.T
        """
        self.X  = matrix( log(X+1) )
        self.XT = matrix(log(X.T+1))
        self.XX = multiply( self.X, self.X )
        self.XXT = multiply( self.XT, self.XT )
        """
        self.nRows, self.nCols = self.X.shape 
        self.nEntries = nnz

class BiclusteringFullData( object ):
    def __init__( self, X, missing = -1, bLog = False, percentTest = 0.0 ):
        self.dim = 1
        if bLog:
            self.X   = log(X+1) 
        else:
            self.X   = X 

        self.nRows, self.nCols = self.X.shape 
        self.nEntries = self.nRows*self.nCols
        self.nMissing = 0
        self.missingval = missing
        self.ProcessForMissing( value = missing )
        self.nTest = int( floor( percentTest*float(self.nEntries - self.nMissing) ) )
        self.XT = self.X.T

        self.NormalizeScale()
        #self.ColNormalize()
        #self.RowNormalize()
        #self.ComboNormalize()
        self.ProcessForTest( value = missing )


    def CorrelationScale(self):
        RC = safecorrcoef(self.X,self.O)
        CC = safecorrcoef(self.XT,self.OT)
        X = zeros( self.X.shape )
        for i,x in zip( range( X.shape[0] ), X ):
            for j in range( X.shape[1] ):
                X[i,j] = RC[i].sum() + CC[j].sum()
        self.X = X
        self.XT = self.X.T
    def ComboNormalize( self ):
        rowmean = zeros( (self.nRows,1), dtype = float )
        for x,i,obs in zip( self.X, range( self.nRows ),self.O ):
            try:
                rowmean[i,0] = x[obs].mean()
            except:
                pass
        colmean = zeros( (1,self.nCols), dtype = float )
        for x,i,obs in zip( self.XT, range( self.nCols ),self.OT ):
            try:
                colmean[0,i] = x[obs].mean()
            except:
                pass
        rowmean /= 2.0
        colmean /= 2.0

        normer =  tile(rowmean,(1,self.nCols)) + tile(colmean,(self.nRows,1))
        figure(2,figsize = (5,5))
        clf()
        ioff()
        matshow(normer,fignum=2,cmap=cm.hot)
        ion() 
        draw()
        """figure(3,figsize = (5,5))
        clf()
        ioff()
        matshow(self.X,fignum=3,cmap=cm.hot)
        ion() 
        draw()"""
        for x,i,obs,nrm in zip( self.X, range( self.nRows ),self.O,normer ):
            try:
                x[obs] -= nrm[obs]
            except:
                pass
        """figure(4,figsize = (5,5))
        clf()
        ioff()
        matshow(self.X,fignum=4,cmap=cm.hot)
        ion() 
        draw()"""
        self.XT = self.X.T

    def RowNormalize( self ):
        for x,i,obs in zip( self.X, range( self.nRows ),self.O ):
            mn = x[obs].mean()
            x[obs] -= mn
        self.XT = self.X.T
        #self.XX  = self.X*self.X 
        #self.XXT = self.XT*self.XT 

    def NormalizeScale( self ):
        minval = inf
        maxval = -inf
        for x,i,obs in zip( self.X, range( self.nRows ),self.O ):
            try:
                mn = x[obs].min()
                mx = x[obs].max()
                if mn < minval:
                    minval = mn
                if mx > maxval:
                    maxval = mx
            except:
                pass
        scaleval = (maxval - minval) / 2.0
        for x,i,obs in zip( self.X, range( self.nRows ),self.O ):
            try:
                x[obs] /= scaleval 
            except:
                pass
            #x[obs] -= 1.0

        self.XT = self.X.T
        #self.XX  = self.X*self.X 
        #self.XXT = self.XT*self.XT 

    def ColNormalize( self ):
        for x,i,obs in zip( self.XT, range( self.nCols ),self.OT ):
            try: 
                mn = x[obs].mean()
                x[obs] -= mn
            except:
                pass
        self.X = self.XT.T
        #self.XX  = self.X*self.X 
        #self.XXT = self.XT*self.XT 

    def ProcessForMissing( self, value = -1 ):
        self.missing = MissingValues(self.nRows,self.nCols)
        self.O = []
        self.Opairs = []
        for x,i in zip( self.X, range( self.nRows ) ):
            newrow = []
            newobs = ones( self.nCols, dtype = bool )
            for xi, j in zip( x, range( self.nCols ) ):
                
                if xi == value:
                    self.nMissing += 1
                    self.missing.AddMissing( i, j )
                    newobs[j] = False
                else:
                    self.Opairs.append( [i,j] )
            self.O.append( newobs )
        self.O  = array( self.O )
        self.OT = self.O.T

        print "MISSING: ",self.nMissing, self.nEntries, float(self.nMissing)/float( self.nEntries )

    def ProcessForTest( self, value = -1 ):
        assert len(self.Opairs) == self.nEntries - self.nMissing, "should match"

        if self.nTest == 0:
            self.OPairs = None
            self.TPairs = None
            self.TestX  = None
            return
        self.testids = random.permutation( len(self.Opairs) )[0:self.nTest]
        self.TPairs  = []
        self.TestX   = [] #array( len(self.TPairs), dtype = float )
        for i in self.testids:
            tpair = self.Opairs[ i ]
            self.TPairs.append( tpair )
            self.TestX.append( self.X[ tpair[0], tpair[1] ] )
            self.missing.AddMissing( tpair[0], tpair[1] )
            self.O[tpair[0]][tpair[1]] = False
            self.X[tpair[0]][tpair[1]] = value
        """for i,x in zip( range( self.nRows), self.X ):
            try:
                col_ids = T[i]
            except:
                col_ids = []
            newobs = zeros( self.nCols, dtype = bool )
            newobs[col_ids] = True
            self.T.append( newobs )
            self.TestX.append( x[col_ids] )
            for j in col_ids:
                self.missing.AddMissing( i, j )
                self.nMissing += 1
            self.O[i][col_ids] = False
            x[col_ids] = value
        """
        #self.T  = array( self.T )
        #self.TT = self.T.T
        self.OPairs = None
        
        self.OT = self.O.T
        print "TEST: ",self.nTest, self.nEntries

    def AddImputedValue( self, i, j, x ):
        self.X[i,j]   = x
        self.XT[j,i]  = x
        #self.XX[i,j]  = x**2
        #self.XXT[j,i] = self.XX[i,j]

class MissingValues( object ):
    def __init__( self,  nRows, nCols  ):
        self.nRows = nRows
        self.nCols = nCols
        self.X     = zeros( ( self.nRows, self.nCols ), dtype = bool )
        self.XT     = zeros( ( self.nCols, self.nRows ), dtype = bool )

    def AddMissing( self, i,j ):
        self.X[i][j]  = 1
        self.XT[j][i] = 1

class BiclusteringFullMultinomialData( object ):
    def __init__( self, X, bLog = False, dim = 1, missing = 999.0 ):
        self.X   = X #log(X+1) 
        self.XT  = self.X.T
        self.dim = dim
        self.nRows, self.nCols = self.X.shape 
        self.nEntries = self.nRows*self.nCols
        self.nMissing = 0
        self.missingval = missing
        self.ProcessForMissing( value = self.missingval )

    def ProcessForMissing( self, value = -1 ):
        self.missing = MissingValues(self.nRows,self.nCols)
        for x,i in zip( self.X, range( self.nRows ) ):
            newrow = []
            for xi, j in zip( x, range( self.nCols ) ):
                if xi == value:
                    self.nMissing += 1
                    self.missing.AddMissing( i, j )

        print "MISSING: ",self.nMissing, self.nEntries, float(self.nMissing)/float( self.nEntries )

    def AddImputedValue( self, i, j, x ):
        self.X[i,j]   = x
        self.XT[j,i]  = x
    """
       Get entry row/column indices for test.
    """
    def RandomlySelectEntries( self, percentTest ):
        self.nTest       = int( percentTest*self.nEntries )
        self.testEntries = random.permutation( self.nEntries )[0:self.nTest] 
        R = zeros( self.nTest, dtype=int )
        C = zeros( self.nTest, dtype=int )
        i = 0
        t = 0
        for i,t in zip( range(self.nTest), self.testEntries ):
           r,c = divmod( t, self.nCols )
           R[i] = r
           C[i] = c
        return R,C