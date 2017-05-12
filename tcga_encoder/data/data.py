from tcga_encoder.utils.helpers import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
import pdb

def load_data_from_dict( y ):      
  y['location'] = os.path.join( os.environ.get('HOME','/'),y['location'])
  y['name_of_store'] = y['name_of_store']
  
  y['dataset'] = MultiSourceData( y['location'], y['name_of_store'] )
  y['store'] = y['dataset'].store
  
def fair_rank( x ):
  #ranks = []
  ix = np.argsort(x)
  sx = x[ix]
  rnk = 0
  old_x = sx[0]
  ranks = [rnk]
  cnter = 1
  for xi in sx[1:]:
    if xi > old_x:
      rnk += 1 #cnter
      cnter = 1
    else:
      cnter += 1
    old_x = xi
    ranks.append(rnk)
  ranks = np.array(ranks, dtype=float)/float(rnk)
  return ranks[np.argsort(ix)]
#
def fair_rank_order_normalization( X ):
  #ORDER = np.argsort( -X, 1 )
  Y = X.copy()
  #vals = np.linspace(eps,1.0-eps,X.shape[1])
  for idx in range(Y.shape[0]):
    Y[idx,:] = fair_rank( Y[idx] )
  return Y
  
def load_gene_filter( gene_filter_name, filter_column, filter_min, sep = "," ):
  print "loading this gene filter file: ", gene_filter_name
  df = pd.read_csv( gene_filter_name, sep = sep )
  if filter_column == "top":
    genes = df["gene"][:filter_min].values
    return genes
  else:
    genes = df["gene"][ df[filter_column] <= filter_min ]
    return genes.values
  return genes
  
class MultiSourceData(object):
  def __init__(self, location, name_of_store = "data" ):
    self.location = location
    self.name_of_store = name_of_store
    
    self.store = self.OpenHdfStore( self.location, self.name_of_store )
    
  def OpenHdfStore(self,location, name_of_store = "data"):
    store_name = "%s.h5"%(name_of_store)
    check_and_mkdir( location )
    full_name = os.path.join( location, store_name )
    
    # I think we can just open in 'a' mode for both
    if os.path.exists(full_name) is False:
      return pd.HDFStore(os.path.join( location, store_name ), "w" )   
    else:
      return pd.HDFStore(os.path.join( location, store_name ), "r+" )    
    
  def CloseHdfStore(self, store):
    return store.close()
   
  # def ReadH5( self, fullpath ):
  #   df = pd.read_hdf( fullpath )
  #   return df
    
  def GetDimension( self, source_name ):
    if source_name == TISSUE:
      return len(self.store[CLINICAL + "/" + TISSUE].columns)
    elif source_name == RNA:
      return len(self.store[RNA + "/" + FAIR].columns)
    elif source_name == miRNA:
      return len(self.store[miRNA + "/" + FAIR].columns)
    elif source_name == DNA:
      return len(self.store[DNA + "/" + CHANNEL + "/0"].columns)
    elif source_name == METH:
      return len(self.store[METH + "/" + FAIR].columns)
    else:
      assert False, "No source called: %s"%(source_name)
      
  def InitInfoFor( self, source_name ):
    self.store[ source_name + "/" + INFO ] = pd.DataFrame( [], columns = ["Name","Value"])
  
  def AddInfo( self, source_name, info_name, info_value ):
    s = pd.Series( [info_name, info_value], index = ["Name","Value"])
    self.store[ source_name + "/" + INFO ] = self.store[ source_name + "/" + INFO ].append( s, ignore_index = True )
  
  def InitSource(self, source_name, broad_location, filename ):
    self.InitInfoFor( source_name )
    self.AddInfo( source_name, "broad_location", broad_location )
    self.AddInfo( source_name, "file", filename )

  def AddClinical( self, broad_location, filename, h5store, diseases = None ):
    print "*****************************************"
    print "**                                     **"
    print "**          CLINICAL                   **"
    print "**                                     **"
    print "*****************************************"
    self.InitSource( CLINICAL, broad_location, filename )
    
    h5 = h5store #self.ReadH5( os.path.join(broad_location, filename) )
    
    if diseases is not None:
      #query = [hd["admin.disease_code"]==d for d in diseases]
      #query=np.array([ (h5["admin.disease_code"]==d).values for d in diseases ]).T.prod(1).astype(bool)
      query=np.squeeze(np.array([ (h5["admin.disease_code"]==d).values.reshape((len(h5),1)) for d in diseases ])).T #.reshape((len(h5),len(diseases))).prod(1).astype(bool)
      #pdb.set_trace()
      #pdb.set_trace()
      h5 = h5[query]
    
    DISEASES = h5["admin.disease_code"].values #.astype(str)
    
    self.clinical_diseases = np.unique( DISEASES  )
    
    self.clinical_disease2idx = OrderedDict()
    self.clinical_barcode2idx = OrderedDict()
    for k,v in zip( self.clinical_diseases, range(len(self.clinical_diseases))):
      self.clinical_disease2idx[ k ] = v
      
    #for k,v in zip( self.clinical_patients, range(len(self.clinical_patients))):
    #  self.clinical_barcode2idx[ k ] = v
      
    PATIENTS = h5["patient.bcr_patient_barcode"].values #.astype(str)
    
    
    PATIENT_INDEX = []


    PATIENTS = DISEASES + "_" + PATIENTS
    self.clinical_patients = np.unique(PATIENTS)
    
    assert len(PATIENTS) == len(DISEASES)
    one_hot_tissue = np.zeros( (len(self.clinical_patients),len(self.clinical_diseases)), dtype=int )
    for idx,barcode,disease in zip(xrange(len(self.clinical_patients)),self.clinical_patients,DISEASES):
      one_hot_tissue[ idx ][self.clinical_disease2idx[disease]] = 1    
    self.store[ CLINICAL + "/" + TISSUE ] = pd.DataFrame( one_hot_tissue, index = self.clinical_patients, columns = self.clinical_diseases )    
    self.store[ CLINICAL + "/" + DATA ] = h5
    self.store[ CLINICAL + "/" + OBSERVED ] = pd.DataFrame( np.ones((len(self.clinical_patients),1),dtype=int), index = self.clinical_patients, columns = [CLINICAL] )
    self.AddObservedPatients( TISSUE, self.clinical_patients )
    #pdb.set_trace()

  def AddObservedPatients( self, source_name, barcodes ):

    print "** Adding observed patients"
    new_df = pd.DataFrame( np.ones((len(barcodes),1),dtype=int), index = barcodes, columns = [source_name] )
    
    self.store[ CLINICAL + "/" + OBSERVED ] = self.store[ CLINICAL + "/" + OBSERVED ].join(new_df, how='outer')
    self.store[ CLINICAL + "/" + OBSERVED ] = self.store[ CLINICAL + "/" + OBSERVED ].fillna(0)

    self.ResolveObservedTissues()

      

  def ResolveObservedTissues(self):
    print "** Resolving observed tissues"
    df = self.store[ CLINICAL + "/" + OBSERVED ]
    t  = self.store[ CLINICAL + "/" + TISSUE ]
    
    observed_barcodes = df.index
    tissue_barcodes = t.index
    missing_bcs = np.setdiff1d( observed_barcodes, tissue_barcodes )
    
    diseases = [dbc.split("_")[0] for dbc in missing_bcs]
    
    for d, dbc in zip( diseases, missing_bcs ):
      #t[d].loc[dbc] = 1
      try:
        self.store[CLINICAL + "/" + TISSUE].loc[dbc]
      except:
        print "** setting patient = %s tissue= %s "%(dbc,d)
        self.store[ CLINICAL + "/" + TISSUE ] = self.store[ CLINICAL + "/" + TISSUE ].set_value(dbc,d,1,)
        self.store[ CLINICAL + "/" + TISSUE ] = self.store[ CLINICAL + "/" + TISSUE ].fillna(0)
        self.store[ CLINICAL + "/" + OBSERVED ] = self.store[ CLINICAL + "/" + OBSERVED ].set_value(dbc,TISSUE,1,)
        self.store[ CLINICAL + "/" + OBSERVED ] = self.store[ CLINICAL + "/" + OBSERVED ].fillna(0)
        try:
          self.store[ CLINICAL + "/" + TISSUE ].loc[dbc]
        except:
          assert False, "Problem assigning to tissue"
        #pdb.set_trace()
    
  def AddDNA( self, broad_location, filename, h5store, h5store_raw, mutation_channels, genes2keep = None, diseases = None, min_nbr_in_pan = None ):
    print "*****************************************"
    print "**                                     **"
    print "**          DNA                        **"
    print "**                                     **"
    print "*****************************************"
    self.InitSource( DNA, broad_location, filename )
    
    h5 = h5store #self.ReadH5( os.path.join(broad_location, filename) )
    h5_raw = h5store_raw 
    h5_merge = h5_raw.append(h5)
    
    duplicate_columns = ["Variant_Classification","patient.bcr_patient_barcode","Hugo_Symbol","Start_Position","End_Position"]
    h5_dropped = h5_merge.drop_duplicates(subset=duplicate_columns)
    #pdb.set_trace()
    #h5.append(h5_raw)
    
    # if diseases is not None:
    #   n=len(h5)
    #   print "** DNA filtering diseases"
    #   query=np.array([ (h5["admin.disease_code"]==d).values.reshape((len(h5),1)) for d in diseases ]).reshape((len(h5),len(diseases))).prod(1).astype(bool)
    #   h5 = h5[query]
    #   n_after = len(h5)
    #   self.AddInfo( DNA, "filtering_step", "disease number%d"%(len(diseases)) )
    #   self.AddInfo( DNA, "filtering_step", "disease filter: from %d to %d"%(n,n_after) )
    
    # figure out patients before filtering out missing genes
    print "finding unique patients"
    PATIENTS = h5["patient.bcr_patient_barcode"].values
    DISEASES = h5["admin.disease_code"].values
    
    u_barcodes = np.sort(np.unique( PATIENTS ) )
    PATIENTS = np.sort(np.unique( DISEASES + "_" + PATIENTS ))
    patient_rows = PATIENTS
    
    raw_PATIENTS = h5_raw["patient.bcr_patient_barcode"].values
    u_raw_barcodes = np.sort(np.unique( raw_PATIENTS ) )
    raw_DISEASES = h5_raw["admin.disease_code"].values
    raw_PATIENTS = np.sort(np.unique( raw_DISEASES + "_" + raw_PATIENTS ))
    raw_patient_rows = raw_PATIENTS

    dropped_PATIENTS = h5_dropped["patient.bcr_patient_barcode"].values
    u_dropped_barcodes = np.sort(np.unique( dropped_PATIENTS ) )
    dropped_DISEASES = h5_dropped["admin.disease_code"].values
    dropped_PATIENTS = np.sort(np.unique( dropped_DISEASES + "_" + dropped_PATIENTS ))
    dropped_patient_rows = dropped_PATIENTS
        
    print "finding intersection of patients" 
    # find patient in h5 that are also in h5_raw, and remove from h5
    intersect_barcodes = np.intersect1d( u_raw_barcodes, u_barcodes)
    intersect_barcodesb = np.intersect1d( u_raw_barcodes, u_dropped_barcodes)
    intersect_barcodesc = np.intersect1d( u_barcodes, u_dropped_barcodes)
    
    print "intersect h5 & raw_h5 ", intersect_barcodes.shape
    print "intersect raw h5 & dropped ", intersect_barcodesb.shape
    print "intersect h5 & dropped ", intersect_barcodesc.shape
    
    #pdb.set_trace()
    # #print "found %d barcodes to remove from h5"%(len(intersect_barcodes))
    # # merge h5 and h5_raw
    #
    # for bc in intersect_barcodes:
    #   n_h5 = len(h5)
    #   #remove_query = np.zeros( (n_h5,1), dtype=bool )
    #   remove_query = (h5["patient.bcr_patient_barcode"]==bc).values.reshape((n_h5,1))
    #   keep_query = ~remove_query
    #   #pdb.set_trace()
    #   h5 = h5[keep_query]
    #
    # new_PATIENTS = h5["patient.bcr_patient_barcode"].values
    # new_DISEASES = h5["admin.disease_code"].values
    #
    # new_u_barcodes = np.sort(np.unique( new_PATIENTS ) )
    # new_PATIENTS = np.sort(np.unique( new_DISEASES + "_" + new_PATIENTS ))
    # new_patient_rows = new_PATIENTS
    #
    #
    # print "finding removing patients from h5"
    #pdb.set_trace()  
    h5 = h5_dropped
    
    PATIENTS = h5["patient.bcr_patient_barcode"].values
    DISEASES = h5["admin.disease_code"].values
    
    u_barcodes = np.sort(np.unique( PATIENTS ) )
    PATIENTS = np.sort(np.unique( DISEASES + "_" + PATIENTS ))
    patient_rows = PATIENTS
    
    self.AddObservedPatients( DNA, patient_rows )
    
    if genes2keep is not None:
      print "** DNA filtering genes"
      n = len(h5)
      g = len(genes2keep)
      pdb.set_trace()
      query = np.squeeze(np.array([ (h5["Hugo_Symbol"]==d).values.reshape((n,1)) for d in genes2keep ])).T.sum(1).astype(bool)   
      h5 = h5[query]
      n_after = len(h5)
      self.AddInfo( DNA, "filtering_step", "genes number%d"%(g) )
      self.AddInfo( DNA, "filtering_step", "genes filter: from %d to %d"%(n,n_after) )
    
    print "** DNA figuring out patients"
    gene_columns = np.sort( np.unique( h5["Hugo_Symbol"].values ) )

    print "** DNA making gene2idx patient2idx"
    self.dna_gene2idx = OrderedDict()
    self.dna_patient2idx = OrderedDict()
    n_genes = len(gene_columns)
    n_patients = len(patient_rows)
    for g,idx in zip( gene_columns, range(n_genes)):
      self.dna_gene2idx[g] = idx
    for g,idx in zip( patient_rows, range(n_patients)):
      self.dna_patient2idx[g] = idx
    
    #pdb.set_trace()
    print "** DNA filtering channels"
    n = len(h5)
    channels = []
    channel_idx = 0
    for mcs in mutation_channels:
      s = ""
      for channel in mcs:
        channels.append(channel)
        s += channel + "_"
      s = s[:-1]
      self.AddInfo( DNA, "channel_%d"%(channel_idx), s )
      channel_idx += 1
    channels = np.array(channels)
    #query = np.squeeze(np.array([ (h5["Variant_Classification"]==d).values.reshape((n,1)) for d in channels ])).T.sum(1).astype(bool)
    
    query = np.zeros( (n,1), dtype=bool )
    for d in channels:
      query |= (h5["Variant_Classification"]==d).values.reshape((n,1))


    #alt_h5 = h5[alt_query]
    h5 = h5[query]
    
    #self.store[ DNA + "/" + MUTATIONS ] = h5
    
    print "** DNA making channel matrices"
    # sparse matrices
    n = len(h5)
    channel_idx = 0
    all_mutations = np.zeros( (n_patients,n_genes), dtype=int )
    for mcs in mutation_channels:
      channels = []
      for channel in mcs:
        channels.append(channel)
        
        query = (h5["Variant_Classification"]==channel).values.reshape((n,1))
        query = query.astype(bool)
        
        variant_mutations = np.zeros( (n_patients,n_genes), dtype=int )
        for disease,barcode, symbol in h5[query][["admin.disease_code","patient.bcr_patient_barcode","Hugo_Symbol"]].values:
          variant_mutations[self.dna_patient2idx[disease+"_"+barcode]][self.dna_gene2idx[symbol]] = 1
          #pdb.set_trace()
      
        all_mutations += variant_mutations
        self.store[ DNA + "/" + VARIANT + "/%s"%channel ] = pd.DataFrame( variant_mutations, index=patient_rows, columns=gene_columns )
        
      
      all_mutations = np.minimum( all_mutations, 1.0 )  
      n_channel = len(channels)
      #query = np.array([ (h5["Variant_Classification"]==d).values.reshape((n,1)) for d in channels ]).reshape((n,n_channel))
      #query = query.sum(1).astype(bool)
      
      query = np.zeros( (n,1), dtype=bool )
      for d in channels:
        query |= (h5["Variant_Classification"]==d).values.reshape((n,1))
       #.T.sum(1).reshape((n,n_channel)).astype(bool)
      
      # query = np.squeeze( query.T.sum(1).reshape((n,n_channel)).astype(bool) )
      
      #h5_channel = h5[query][["patient.bcr_patient_barcode","Hugo_symbol"]].values
      
      channel_mutations = np.zeros( (n_patients,n_genes), dtype=int )
      for disease, barcode, symbol in h5[query][["admin.disease_code","patient.bcr_patient_barcode","Hugo_Symbol"]].values:
        channel_mutations[self.dna_patient2idx[disease+"_"+barcode]][self.dna_gene2idx[symbol]] = 1

      # alt_channel_mutations = np.zeros( (n_patients,n_genes), dtype=int )
      # for disease, barcode, symbol in h5[alt_query][["admin.disease_code","patient.bcr_patient_barcode","Hugo_Symbol"]].values:
      #   alt_channel_mutations[self.dna_patient2idx[disease+"_"+barcode]][self.dna_gene2idx[symbol]] += 1
      
      
      print "query.sum() = %d"%(query.sum())
      #print "alt_query.sum() = %d"%(alt_query.sum())
      print "Channel mutations = %d"%(channel_mutations.sum())
      #print "AltChannel mutations = %d"%(alt_channel_mutations.sum())
      print "All     mutations = %d"%(all_mutations.sum())
      #pdb.set_trace()
      assert channel_mutations.sum() == all_mutations.sum(), "should be the same"
      self.store[ DNA + "/" + CHANNEL + "/%d"%channel_idx ] = pd.DataFrame( channel_mutations, index=patient_rows, columns=gene_columns )
      
      if min_nbr_in_pan is not None:
        summed = self.store[ DNA + "/" + CHANNEL + "/%d"%channel_idx ].sum()
        selected_genes = summed[summed>=min_nbr_in_pan].index
        self.store[ DNA + "/" + CHANNEL + "/%d"%channel_idx ] = self.store[ DNA + "/" + CHANNEL + "/%d"%channel_idx ][ selected_genes ]
        #pdb.set_trace()
      print self.store[ DNA + "/" + CHANNEL + "/%d"%channel_idx ].sum().sort_values(ascending=False)
      channel_idx+=1  
      
  def AddRNA( self, broad_location, filename, h5store_ga, h5store_hi, nbr_genes, method = "max_var_fair", diseases = None ):
    #genes2keep = None, diseases = None ):
    print "*****************************************"
    print "**                                     **"
    print "**          RNA                        **"
    print "**                                     **"
    print "*****************************************"

    self.InitSource( RNA, broad_location, filename )
    
    h5store_ga, h5store_hi
    
    h5_b = h5store_hi.append(h5store_ga)
    h5 = h5_b.drop_duplicates( subset=["RNApatient.bcr_patient_barcode"])
    
    # h5_barcodes = h5["patient.bcr_patient_barcode"].values
    # u_h5_barcodes = np.unique(h5_barcodes)
    #
    # counter_ga = Counter(h5store_ga["RNApatient.bcr_patient_barcode"].values)
    # counter_hi = Counter(h5store_hi["RNApatient.bcr_patient_barcode"].values)
    #
    # counts_ga = np.array(counter_ga.values())
    # I_ga = pp.find( counts_ga > 1 )
    #
    # dup_bcs_ga = np.array( counter_ga.keys() )[I_ga]
    # counts_hi = np.array(counter_hi.values())
    # I_hi = pp.find( counts_hi > 1 )
    #
    # dup_bcs_hi = np.array( counter_hi.keys() )[I_hi]
    
    #pdb.set_trace()
    #h5 = h5store #self.ReadH5( os.path.join(broad_location, filename) )
    
    # if diseases is not None:
    #   n=len(h5)
    #   print "** RNA filtering diseases"
    #   index_array = np.array(h5.index.values)
    #   query=np.array([ (h5["admin.disease_code"]==d).values.reshape((len(h5),1)) for d in diseases ]).reshape((len(h5),len(diseases))).prod(1).astype(bool)
    #   h5 = h5[query]
    #   n_after = len(h5)
    #   self.AddInfo( RNA, "filtering_step", "disease number%d"%(len(diseases)) )
    #   self.AddInfo( RNA, "filtering_step", "disease filter: from %d to %d"%(n,n_after) )
    
    #gene_columns = np.sort( np.unique( h5["Hugo_Symbol"].values ) )
    print "** RNA filter for tumor samples only"
    
    patient_disease = h5["admin.disease_code"].values
    patient_bcs = h5["patient.bcr_patient_barcode"].values
    patient_rows = h5["RNApatient.bcr_patient_barcode"].values
    keep_bcs = []
    keep_query = []
    for disease,bc,pbc in zip(patient_disease,patient_rows,patient_bcs):
      #if pbc=="tcga-vq-a8p8":
      #  print disease,pbc, bc
      if bc[13:15] == '01' and bc[-2:] != "_x" and bc[-2:] != "_y":
        assert bc[:12] == pbc, "these should be the same"
        keep_bcs.append(disease+"_"+pbc)
        keep_query.append(True)
      else:
        keep_query.append(False)
    keep_bcs = np.array(keep_bcs)
    keep_query = np.array(keep_query)    
    
    assert len(keep_bcs) == len(np.unique(keep_bcs)), "should be unique list"
    h5 = h5[keep_query]
    patient_rows = patient_disease[keep_query]+"_"+patient_bcs[keep_query] #h5["patient.bcr_patient_barcode"].values
    
    #pdb.set_trace()
    self.AddObservedPatients( RNA, patient_rows )
    
      
    #self.rna_h5 = h5
    
    print "** RNA splitting genes"
    self.rna_original_genes = h5.columns
    self.rna_original2index = OrderedDict()
    self.rna_hugo2index = OrderedDict()
    for k,v in zip( self.rna_original_genes, xrange(len(self.rna_original_genes))):
      if k == "admin.disease_code" or k == "patient.bcr_patient_barcode" or k == "RNApatient.bcr_patient_barcode":
        continue
      self.rna_original2index[k] = v
      hugo,entrez = k.split("|")
      self.rna_hugo2index[hugo] = v
    
    genes2keep = None
    if genes2keep is not None:
      self.rna_genes2keep2idx = OrderedDict()
      for g in genes2keep:
        if self.rna_hugo2index.has_key(g):
          self.rna_genes2keep2idx[g] = self.rna_hugo2index[g]
      
      gene_order = np.argsort( self.rna_genes2keep2idx.keys() )
      gene_columns = np.array(self.rna_genes2keep2idx.keys())[gene_order]
      gene_ids = np.array(self.rna_genes2keep2idx.values())[gene_order]
    else:
      gene_order = np.argsort( self.rna_hugo2index.keys() )
      gene_columns = np.array(self.rna_hugo2index.keys())[gene_order]
      gene_ids = np.array(self.rna_hugo2index.values())[gene_order]
      
    
    R =   h5.values[:, gene_ids ].astype(float)
    

    I = pp.find( np.isnan(R.sum(0) )==False )
    R = R[:,I]
    gene_columns = gene_columns[I]
    FAIR_R = fair_rank_order_normalization(R)
    
    if method == "max_var_fair":
      v = np.var( FAIR_R, 0 )
      gene_ids = np.argsort( v )[-nbr_genes:]
      gene_columns = gene_columns[gene_ids]
      
      I = np.argsort( gene_columns )
      gene_ids = gene_ids[I]
      gene_columns = gene_columns[I]
      
      FAIR_R = FAIR_R[:,gene_ids]
      R = R[:,gene_ids]
      #pdb.set_trace()
    elif method is None:
      pass
    elif method == "none":
      pass
    else:
      assert False, "unknown selection method for RNA = %s"%(method)
    self.store[ RNA + "/" + "RSEM" + "/" ] = pd.DataFrame( R, index = patient_rows, columns = gene_columns )
    self.store[ RNA + "/" + "FAIR" + "/" ] = pd.DataFrame( FAIR_R, index = patient_rows, columns = gene_columns )

    
  def AddMeth( self, broad_location, filename, h5store, nbr_genes, method = "max_var_fair", diseases = None ):
    print "*****************************************"
    print "**                                     **"
    print "**          METHYLATION                **"
    print "**                                     **"
    print "*****************************************"
    self.InitSource( METH, broad_location, filename )
    
    h5 = h5store #self.ReadH5( os.path.join(broad_location, filename) )
    
    if diseases is not None:
      n=len(h5)
      print "** METH filtering diseases"
      index_array = np.array(h5.index.values)
      query=np.array([ (h5["admin.disease_code"]==d).values.reshape((len(h5),1)) for d in diseases ]).reshape((len(h5),len(diseases))).prod(1).astype(bool)
      h5 = h5[query]
      n_after = len(h5)
      self.AddInfo( METH, "filtering_step", "disease number%d"%(len(diseases)) )
      self.AddInfo( METH, "filtering_step", "disease filter: from %d to %d"%(n,n_after) )
    
    print "** METH filter for tumor samples only"
    patient_disease = h5["admin.disease_code"].values
    patient_bcs = h5["patient.bcr_patient_barcode"].values
    patient_rows = h5["Methpatient.bcr_patient_barcode"].values
    keep_bcs = []
    keep_query = []
    for disease,bc,pbc in zip(patient_disease,patient_rows,patient_bcs):
      if pbc=="tcga-vq-a8p8":
        print disease,pbc, bc
      if bc[13:15] == '01':
        assert bc[:12] == pbc, "these should be the same"
        keep_bcs.append(disease+"_"+pbc)
        keep_query.append(True)
      else:
        keep_query.append(False)
    keep_bcs = np.array(keep_bcs)
    keep_query = np.array(keep_query)    
    
    assert len(keep_bcs) == len(np.unique(keep_bcs)), "should be unique list"
    h5 = h5[keep_query]
    patient_rows = patient_disease[keep_query]+"_"+patient_bcs[keep_query]#h5["patient.bcr_patient_barcode"].values
    
    self.AddObservedPatients( METH, patient_rows )
    
    print "** METH splitting genes"
    self.meth_original_genes = h5.columns
    self.meth_original2index = OrderedDict()
    self.meth_hugo2index = OrderedDict()
    for k,v in zip( self.meth_original_genes, xrange(len(self.meth_original_genes))):
      if k == "admin.disease_code" or k == "patient.bcr_patient_barcode" or k == "Methpatient.bcr_patient_barcode":
        continue
      self.meth_original2index[k] = v
      #hugo,entrez = k.split("|")
      self.meth_hugo2index[k] = v
    
    genes2keep=None
    if genes2keep is not None:
      self.meth_genes2keep2idx = OrderedDict()
      for g in genes2keep:
        if self.meth_hugo2index.has_key(g):
          self.meth_genes2keep2idx[g] = self.meth_hugo2index[g]
      
      gene_order = np.argsort( self.meth_genes2keep2idx.keys() )
      gene_columns = np.array(self.meth_genes2keep2idx.keys())[gene_order]
      gene_ids = np.array(self.meth_genes2keep2idx.values())[gene_order]
    else:
      gene_order = np.argsort( self.meth_hugo2index.keys() )
      gene_columns = np.array(self.meth_hugo2index.keys())[gene_order]
      gene_ids = np.array(self.meth_hugo2index.values())[gene_order]
      
    
    R =   h5.values[:, gene_ids ].astype(float)
    
    print "** METH cleaning Nan genes"
    self.AddInfo( METH, "filtering_step", "METH had %d genes"%(len(gene_ids)) )
    I = pp.find( np.isnan(R.sum(0) )==False )
    R = R[:,I]
    gene_columns = gene_columns[I]
    self.AddInfo( METH, "filtering_step", "METH has %d genes"%(len(I)) )
    
    FAIR_R = fair_rank_order_normalization(R)
    
    if method == "max_var_fair":
      v = np.var( FAIR_R, 0 )
      gene_ids = np.argsort( v )[-nbr_genes:]
      gene_columns = gene_columns[gene_ids]
      
      I = np.argsort( gene_columns )
      gene_ids = gene_ids[I]
      gene_columns = gene_columns[I]
      FAIR_R = FAIR_R[:,gene_ids]
      R = R[:,gene_ids]
      
      
      #pdb.set_trace()
    elif method is None:
      pass
    elif method == "none":
      pass
    else:
      assert False, "unknown selection method for RNA = %s"%(method)
    
    self.AddInfo( METH, "filtering_step", "METH has %d genes"%(len(gene_columns)) )
    
    
    
    
    
    self.store[ METH + "/" + "METH" + "/" ] = pd.DataFrame( R, index = patient_rows, columns = gene_columns )
    self.store[ METH + "/" + "FAIR" + "/" ] = pd.DataFrame( FAIR_R, index = patient_rows, columns = gene_columns )
    
    
    #self.meth_h5 = h5
  def AddmiRNA( self, broad_location, filename, h5store_ga, h5store_hi, nbr_hsas, method = "max_var_fair", diseases = None ):
    #genes2keep = None, diseases = None ):
    print "*****************************************"
    print "**                                     **"
    print "**       mi RNA                        **"
    print "**                                     **"
    print "*****************************************"

    self.InitSource( miRNA, broad_location, filename )
    
    h5_b = h5store_hi.append(h5store_ga)
    h5 = h5_b.drop_duplicates( subset=["miRNApatient.bcr_patient_barcode"])
    #h5 = h5store #self.ReadH5( os.path.join(broad_location, filename) )
    
    if diseases is not None:
      n=len(h5)
      print "** miRNA filtering diseases"
      index_array = np.array(h5.index.values)
      query=np.array([ (h5["admin.disease_code"]==d).values.reshape((len(h5),1)) for d in diseases ]).reshape((len(h5),len(diseases))).prod(1).astype(bool)
      h5 = h5[query]
      n_after = len(h5)
      self.AddInfo( miRNA, "filtering_step", "disease number%d"%(len(diseases)) )
      self.AddInfo( miRNA, "filtering_step", "disease filter: from %d to %d"%(n,n_after) )
    
    #gene_columns = np.sort( np.unique( h5["Hugo_Symbol"].values ) )
    print "** miRNA filter for tumor samples only"
    
    patient_disease = h5["admin.disease_code"].values
    patient_bcs = h5["patient.bcr_patient_barcode"].values
    patient_rows = h5["miRNApatient.bcr_patient_barcode"].values
    keep_bcs = []
    keep_query = []
    for disease,bc,pbc in zip(patient_disease,patient_rows,patient_bcs):
      assert bc[:12] == pbc, "these should be the same"
      keep_bcs.append(disease+"_"+pbc)
      keep_query.append(True)
      
    keep_bcs = np.array(keep_bcs)
    keep_query = np.array(keep_query)    
    
    assert len(keep_bcs) == len(np.unique(keep_bcs)), "should be unique list"
    h5 = h5[keep_query]
    patient_rows = patient_disease[keep_query]+"_"+patient_bcs[keep_query] #h5["patient.bcr_patient_barcode"].values
    
    #pdb.set_trace()
    self.AddObservedPatients( miRNA, patient_rows )
    
      
    #self.rna_h5 = h5
    
    print "** miRNA splitting HSA"
    self.mirna_original_hsas = h5.columns
    self.mirna_original2index = OrderedDict()
    self.mirna_hsa2index = OrderedDict()
    for k,v in zip( self.mirna_original_hsas, xrange(len(self.mirna_original_hsas))):
      if k == "admin.disease_code" or k == "patient.bcr_patient_barcode" or k == "miRNApatient.bcr_patient_barcode":
        continue
      self.mirna_original2index[k] = v
      #hugo,entrez = k.split("|")
      self.mirna_hsa2index[k] = v
    
    hsas2keep = None
    if hsas2keep is not None:
      self.mirna_hsas2keep2idx = OrderedDict()
      for g in hsas2keep:
        if self.mirna_hsa2index.has_key(g):
          self.mirna_hsas2keep2idx[g] = self.mirna_hsa2index[g]
      
      hsa_order = np.argsort( self.mirna_hsas2keep2idx.keys() )
      hsa_columns = np.array(self.mirna_hsas2keep2idx.keys())[hsa_order]
      hsa_ids = np.array(self.mirna_hsas2keep2idx.values())[hsa_order]
    else:
      hsa_order = np.argsort( self.mirna_hsa2index.keys() )
      hsa_columns = np.array(self.mirna_hsa2index.keys())[hsa_order]
      hsa_ids = np.array(self.mirna_hsa2index.values())[hsa_order]
      
    
    R =   h5.values[:, hsa_ids ].astype(float)
    
    #pdb.set_trace()
    I = pp.find( np.isnan(R.sum(0) )==False )
    R = R[:,I]
    hsa_columns = hsa_columns[I]
    FAIR_R = fair_rank_order_normalization(R)
    
    if method == "max_var_fair":
      v = np.var( FAIR_R, 0 )
      hsa_ids = np.argsort( v )[-nbr_hsas:]
      hsa_columns = hsa_columns[hsa_ids]
      
      I = np.argsort( hsa_columns )
      hsa_ids = hsa_ids[I]
      hsa_columns = hsa_columns[I]
      
      FAIR_R = FAIR_R[:,hsa_ids]
      R = R[:,hsa_ids]
      #pdb.set_trace()
    elif method is None:
      pass
    elif method == "none":
      pass
    else:
      assert False, "unknown selection method for RNA = %s"%(method)
    self.store[ miRNA + "/" + "READS" + "/" ] = pd.DataFrame( R, index = patient_rows, columns = hsa_columns )
    self.store[ miRNA + "/" + "FAIR" + "/" ] = pd.DataFrame( FAIR_R, index = patient_rows, columns = hsa_columns )    
    #pdb.set_trace()
    
    
    
    
    
    
    
    
    