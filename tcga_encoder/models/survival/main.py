from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.models.survival_analysis import *
#from tcga_encoder.algorithms import *
import seaborn as sns
import pdb

sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(yaml_file):
  y = load_yaml( yaml_file)
  load_data_from_dict( y[DATA] )
  data_dict      = y[DATA] #{N_TRAIN:4000}
  survival_dict  = y["survival"]
  logging_dict   = y[LOGGING]
  
  logging_dict[SAVEDIR] = os.path.join( HOME_DIR, os.path.join( logging_dict[LOCATION], logging_dict[EXPERIMENT] ) )

  
  #data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
  fill_location = os.path.join( logging_dict[SAVEDIR], "full_vae_fill.h5" )
  survival_location = os.path.join( logging_dict[SAVEDIR], "full_vae_survival.h5" )
  #savename = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/tiny/leave_out_%s/survival_xval.png"%(disease))
  
  print "FILL: ", fill_location
  print "SURV: ", survival_location
  s=pd.HDFStore( survival_location, "r" )
  d=data_dict['store'] #pd.HDFStore( data_location, "r" )
  f=pd.HDFStore( fill_location, "r" ) 
  
  #pdb.set_trace()
  
  for survival_spec in survival_dict:
    name = survival_spec["name"]
    if name == "lda_xval":
      print "running LDA"
      folds = survival_spec["folds"]
      bootstraps = survival_spec["bootstraps"]
      epsilon =  survival_spec["epsilon"]
      save_location = os.path.join( logging_dict[SAVEDIR], "survival_lda_xval.png" )  
      projections, probabilties, weights, averages, X, y, E_train, T_train = run_survival_analysis_lda( data_dict['validation_tissues'], f, d, k_fold = folds, n_bootstraps = bootstraps, epsilon= epsilon )  
    
      avg_proj = averages[0]
      avg_prob = averages[1]
  
      fig = pp.figure()
      mn_proj = projections[0]
      std_proj = np.sqrt(projections[1])
      mn_prob = probabilties[0]
      std_prob = np.sqrt(probabilties[1])
      mn_w = weights[0]
      std_w = np.sqrt(weights[1])
  
      ax1 = fig.add_subplot(111)
      I = pp.find( np.isnan(mn_prob))
      mn_prob[I] = 0
      I = pp.find( np.isinf(mn_prob))
      mn_prob[I] = 1
      
      I = pp.find( np.isnan(mn_proj))
      mn_proj[I] = 0
      I = pp.find( np.isinf(mn_proj))
      mn_proj[I] = 1
      
      
      #I = np.argsort(-mn_proj)
      I = np.argsort(-mn_prob)
      half = int(len(I)/2.0)
      I0 = I[:half]
      I1 = I[half:]
      kmf = KaplanMeierFitter()
      if len(I1) > 0:
        kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
      if len(I0) > 0:
        kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
      pp.savefig(save_location, dpi=300, format='png')
      print "ROC mn_prob ", roc_auc_score(y,mn_prob)
      print "ROC mn_proj ", roc_auc_score(y,mn_proj)
    
    elif name == "lda_loo":
      print "running LDA loo"
      folds = survival_spec["folds"]
      bootstraps = survival_spec["bootstraps"]
      epsilon =  survival_spec["epsilon"]
      save_location = os.path.join( logging_dict[SAVEDIR], "survival_lda_loo.png" )  
      projections, probabilties, weights, averages, X, y, E_train, T_train = run_survival_analysis_lda( data_dict['validation_tissues'], f, d, k_fold = folds, n_bootstraps = bootstraps, epsilon= epsilon )  
    
      avg_proj = averages[0]
      avg_prob = averages[1]
  
      fig = pp.figure()
      mn_proj = projections[0]
      std_proj = np.sqrt(projections[1])
      mn_prob = probabilties[0]
      std_prob = np.sqrt(probabilties[1])
      mn_w = weights[0]
      std_w = np.sqrt(weights[1])
  
      ax1 = fig.add_subplot(111)
      I = pp.find( np.isnan(mn_prob))
      mn_prob[I] = 0
      I = pp.find( np.isinf(mn_prob))
      mn_prob[I] = 1
      
      I = pp.find( np.isnan(mn_proj))
      mn_proj[I] = 0
      I = pp.find( np.isinf(mn_proj))
      mn_proj[I] = 1
      
      
      #I = np.argsort(-mn_proj)
      I = np.argsort(-mn_prob)
      half = int(len(I)/2.0)
      I0 = I[:half]
      I1 = I[half:]
      kmf = KaplanMeierFitter()
      if len(I1) > 0:
        kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
      if len(I0) > 0:
        kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
      pp.savefig(save_location, dpi=300, format='png')
      print "ROC mn_prob ", roc_auc_score(y,mn_prob)
      print "ROC mn_proj ", roc_auc_score(y,mn_proj)

    elif name == "lda_train":
      print "running LDA"
      folds = survival_spec["folds"]
      bootstraps = survival_spec["bootstraps"]
      epsilon =  survival_spec["epsilon"]
      save_location = os.path.join( logging_dict[SAVEDIR], "survival_lda_train.png" )  
      projections, probabilties, weights, averages, X, y, E_train, T_train = run_survival_analysis_lda_train( data_dict['validation_tissues'], f, d, k_fold = folds, n_bootstraps = bootstraps, epsilon= epsilon )  
    
      avg_proj = averages[0]
      avg_prob = averages[1]
  
      fig = pp.figure()
      mn_proj = projections[0]
      std_proj = np.sqrt(projections[1])
      mn_prob = probabilties[0]
      std_prob = np.sqrt(probabilties[1])
      mn_w = weights[0]
      std_w = np.sqrt(weights[1])
  
      ax1 = fig.add_subplot(111)
      I = pp.find( np.isnan(mn_prob))
      mn_prob[I] = 0
      I = pp.find( np.isinf(mn_prob))
      mn_prob[I] = 1
      
      I = pp.find( np.isnan(mn_proj))
      mn_proj[I] = 0
      I = pp.find( np.isinf(mn_proj))
      mn_proj[I] = 1
      
      
      #I = np.argsort(-mn_proj)
      I = np.argsort(-mn_prob)
      half = int(len(I)/2.0)
      I0 = I[:half]
      I1 = I[half:]
      kmf = KaplanMeierFitter()
      if len(I1) > 0:
        kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
      if len(I0) > 0:
        kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
      pp.savefig(save_location, dpi=300, format='png')
      print "ROC mn_prob ", roc_auc_score(y,mn_prob)
      print "ROC mn_proj ", roc_auc_score(y,mn_proj)    
    
    elif name == "kmeans":
      print "running kmeans"
      folds = survival_spec["folds"]
      K = survival_spec["K"]
      predictions, X, y, E_train, T_train = run_survival_analysis_kmeans( data_dict['validation_tissues'], f, d, k_fold = folds, K = K )  
      
      save_location = os.path.join( logging_dict[SAVEDIR], "survival_kmeans.png" )
      fig = pp.figure()
      kmf = KaplanMeierFitter()
      ax1 = fig.add_subplot(111)
      
      colours = "brgkmcbrgkmcbrgkmcbrgkmcbrgkmcbrgkmcbrgkmc"
      for k in range(K):
        I = pp.find( predictions==k)
        Ti=T_train[I]
        Ei=E_train[I]

        if len(Ti)>0:
          kmf.fit(Ti, event_observed=Ei, label =  "k =%d E=%d C=%d"%(k,E_train[I].sum(),len(I)-E_train[I].sum()) )
          ax1=kmf.plot(ax=ax1, at_risk_counts=False,show_censors=True, color=colours[k])
          

      
      pp.savefig( save_location, fmt='png', dpi=300 )
      #bootstraps = survival_spec["bootstraps"]
      
  
  
######################################################################################################
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file  
  main( yaml_file )

  
  
  