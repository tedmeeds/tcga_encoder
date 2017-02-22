from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.models.survival_analysis import *
#from tcga_encoder.algorithms import *
import seaborn as sns
import pdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(yaml_file, weights_matrix):
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
      third = int(len(I)/3.0)
      I0 = I[:third]
      I1 = I[third:2*third]
      I2 = I[2*third:]
      kmf = KaplanMeierFitter()
      if len(I2) > 0:
        kmf.fit(T_train[I2], event_observed=E_train[I2], label =  "lda_1 E=%d C=%d"%(E_train[I2].sum(),len(I2)-E_train[I2].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
      if len(I1) > 0:
        kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='green')
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
      save_weights_template = os.path.join( logging_dict[SAVEDIR], "survival_weights_" ) 
      projections, probabilties, weights, averages, X, y, E_train, T_train = run_survival_analysis_lda_loo( data_dict['validation_tissues'], f, d, k_fold = folds, n_bootstraps = bootstraps, epsilon= epsilon )  
      disease = data_dict['validation_tissues'][0]
      
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
      
      survival_target = -mn_proj
      C = 1.0
      C_reg = 0.0045
      n2show = 10
      reg_rna_class_projections, \
      reg_rna_class_weights, \
      reg_rna_class_averages, \
      reg_rna_class_X, \
      reg_rna_class_y = run_survival_prediction_loo_regression( data_dict['validation_tissues'], f, d, targets=survival_target, data_keys = ["/RNA/FAIR"], data_names = ["RNA"], C = C_reg )  
      
      reg_rna_Ws = reg_rna_class_weights[2]
      reg_rna_bs = reg_rna_class_weights[3] 
      print 'biases: ', reg_rna_bs.mean()
      I_reg_rna_parameters = np.argsort( -np.abs( reg_rna_class_weights[0] ) )
      I_reg_rna_predictions = np.argsort( reg_rna_class_projections[0] )
      
      print "reg RNA ranked high, ", [reg_rna_class_X.columns[i] for i in I_reg_rna_parameters[:n2show]] 
      #print "reg RNA ranked low, ", [reg_rna_class_X.columns[i] for i in I_reg_rna_parameters[-n2show:]] 
      
      
      #pp.show()
      
      y_pred_rna = reg_rna_class_projections[0]; 
      y_true_rna = reg_rna_class_y ; #y_true -= y_true.mean() 
      reg_rna_r2 = r2_score(y_true_rna, y_pred_rna)  
      reg_rna_mse = mean_squared_error(y_true_rna, y_pred_rna)  
      reg_rna_expvar = explained_variance_score(y_true_rna, y_pred_rna)  
      reg_rna_corr = np.corrcoef( y_true_rna, y_pred_rna)[0][1]

      #--------
      reg_meth_class_projections, \
      reg_meth_class_weights, \
      reg_meth_class_averages, \
      reg_meth_class_X, \
      reg_meth_class_y = run_survival_prediction_loo_regression( data_dict['validation_tissues'], f, d, targets=survival_target, data_keys = ["/METH/FAIR"], data_names = ["METH"], C = C_reg )  
      
      reg_meth_Ws = reg_meth_class_weights[2]
      reg_meth_bs = reg_meth_class_weights[3] 
      print 'biases: ', reg_meth_bs.mean()
      I_reg_meth_parameters = np.argsort( -np.abs( reg_meth_class_weights[0] ) )
      I_reg_meth_predictions = np.argsort( reg_meth_class_projections[0] )
      print "reg METH ranked high, ", [reg_meth_class_X.columns[i] for i in I_reg_meth_parameters[:n2show]] 
      #print "reg METH ranked low, ", [reg_meth_class_X.columns[i] for i in I_reg_meth_parameters[-n2show:]] 
      
      
      #pp.show()
      
      y_pred_meth = reg_meth_class_projections[0]; 
      y_true_meth = reg_meth_class_y ; #y_true -= y_true.mean() 
      reg_meth_r2 = r2_score(y_true_meth, y_pred_meth)  
      reg_meth_mse = mean_squared_error(y_true_meth, y_pred_meth)  
      reg_meth_expvar = explained_variance_score(y_true_meth, y_pred_meth)  
      reg_meth_corr = np.corrcoef( y_true_meth, y_pred_meth)[0][1]
      # -----------  

      #--------
      reg_dna_class_projections, \
      reg_dna_class_weights, \
      reg_dna_class_averages, \
      reg_dna_class_X, \
      reg_dna_class_y = run_survival_prediction_loo_regression( data_dict['validation_tissues'], f, d, targets=survival_target, data_keys = ["/DNA/channel/0"], data_names = ["DNA"], C = C_reg )  
      
      reg_dna_Ws = reg_dna_class_weights[2]
      reg_dna_bs = reg_dna_class_weights[3] 
      print 'biases: ', reg_dna_bs.mean()
      I_reg_dna_parameters = np.argsort( -np.abs( reg_dna_class_weights[0] ) )
      I_reg_dna_predictions = np.argsort( reg_dna_class_projections[0] )
      print "reg DNA ranked high, ", [reg_dna_class_X.columns[i] for i in I_reg_dna_parameters[:n2show]] 
      #print "reg DNA ranked low, ", [reg_dna_class_X.columns[i] for i in I_reg_dna_parameters[-n2show:]] 
      
      
      #pp.show()
      
      y_pred_dna = reg_dna_class_projections[0]; 
      y_true_dna = reg_dna_class_y ; #y_true -= y_true.mean() 
      reg_dna_r2 = r2_score(y_true_dna, y_pred_dna)  
      reg_dna_mse = mean_squared_error(y_true_dna, y_pred_dna)  
      reg_dna_expvar = explained_variance_score(y_true_dna, y_pred_dna)  
      reg_dna_corr = np.corrcoef( y_true_dna, y_pred_dna)[0][1]
      # -----------

      #--------
      reg_combo_class_projections, \
      reg_combo_class_weights, \
      reg_combo_class_averages, \
      reg_combo_class_X, \
      reg_combo_class_y = run_survival_prediction_loo_regression( data_dict['validation_tissues'], f, d, targets=survival_target, data_keys = ["/DNA/channel/0","/RNA/FAIR","/METH/FAIR"], data_names = ["DNA","RNA","METH"], C = C_reg )  
      
      reg_combo_Ws = reg_combo_class_weights[2]
      reg_combo_bs = reg_combo_class_weights[3] 
      print 'biases: ', reg_combo_bs.mean()
      I_reg_combo_parameters = np.argsort( -np.abs( reg_combo_class_weights[0] ) )
      I_reg_combo_predictions = np.argsort( reg_combo_class_projections[0] )
      print "reg COMBO ranked high, ", [reg_combo_class_X.columns[i] for i in I_reg_combo_parameters[:n2show]] 
      #print "reg COMBO ranked low, ", [reg_combo_class_X.columns[i] for i in I_reg_combo_parameters[-n2show:]] 
      pd.DataFrame( reg_combo_class_weights[0][I_reg_combo_parameters], columns=["%s_COMB_reg"%disease],index = [reg_combo_class_X.columns[i] for i in I_reg_combo_parameters] ).to_csv(save_weights_template+"combo_reg.csv")
      pd.DataFrame( reg_dna_class_weights[0][I_reg_dna_parameters], columns=["%s_DNA_reg"%disease],index = [reg_dna_class_X.columns[i] for i in I_reg_dna_parameters] ).to_csv(save_weights_template+"dna_reg.csv")
      pd.DataFrame( reg_meth_class_weights[0][I_reg_meth_parameters], columns=["%s_METH_reg"%disease],index = [reg_meth_class_X.columns[i] for i in I_reg_meth_parameters] ).to_csv(save_weights_template+"meth_reg.csv")
      pd.DataFrame( reg_rna_class_weights[0][I_reg_rna_parameters], columns=["%s_RNA_reg"%disease],index = [reg_rna_class_X.columns[i] for i in I_reg_rna_parameters] ).to_csv(save_weights_template+"rna_reg.csv")
      
      pd.DataFrame( reg_combo_class_weights[0][I_reg_combo_parameters], columns=["%s_COMB_reg"%disease],index = [reg_combo_class_X.columns[i] for i in I_reg_combo_parameters] ).to_csv(save_weights_template+"combo_reg.txt", sep=" " )
      pd.DataFrame( reg_dna_class_weights[0][I_reg_dna_parameters], columns=["%s_DNA_reg"%disease],index = [reg_dna_class_X.columns[i] for i in I_reg_dna_parameters] ).to_csv(save_weights_template+"dna_reg.txt", sep=" ")
      pd.DataFrame( reg_meth_class_weights[0][I_reg_meth_parameters], columns=["%s_METH_reg"%disease],index = [reg_meth_class_X.columns[i] for i in I_reg_meth_parameters] ).to_csv(save_weights_template+"meth_reg.txt", sep=" ")
      pd.DataFrame( reg_rna_class_weights[0][I_reg_rna_parameters], columns=["%s_RNA_reg"%disease],index = [reg_rna_class_X.columns[i] for i in I_reg_rna_parameters] ).to_csv(save_weights_template+"rna_reg.txt", sep=" ")
      
      #pp.show()
      
      y_pred_combo = reg_combo_class_projections[0]; 
      y_true_combo = reg_combo_class_y ; #y_true -= y_true.mean() 
      reg_combo_r2 = r2_score(y_true_combo, y_pred_combo)  
      reg_combo_mse = mean_squared_error(y_true_combo, y_pred_combo)  
      reg_combo_expvar = explained_variance_score(y_true_combo, y_pred_combo)  
      reg_combo_corr = np.corrcoef( y_true_combo, y_pred_combo)[0][1]
      # -----------
                
      save_location_reg = os.path.join( logging_dict[SAVEDIR], "survival_predictions_reg_W.png" )  
      f0=pp.figure()
      ax1 = f0.add_subplot(321)
      ax1.plot(reg_dna_Ws.T, 'k.', alpha=0.5)
      ax2 = f0.add_subplot(322)
      ax2.plot( reg_dna_Ws.T.mean(1), reg_dna_class_X.values.mean(0), '.', alpha=0.5)

      ax1 = f0.add_subplot(323)
      ax1.plot(reg_rna_Ws.T, 'k.', alpha=0.5)
      ax2 = f0.add_subplot(324)
      ax2.plot( reg_rna_Ws.T.mean(1), reg_rna_class_X.values.mean(0), '.', alpha=0.5)


      ax1 = f0.add_subplot(325)
      ax1.plot(reg_meth_Ws.T, 'k.', alpha=0.5)
      ax2 = f0.add_subplot(326)
      ax2.plot( reg_meth_Ws.T.mean(1), reg_meth_class_X.values.mean(0), '.', alpha=0.5)

      pp.savefig(save_location_reg, dpi=300, format='png')
      
      
      print "reg RNA r2     = ", reg_rna_r2
      print "reg RNA mse    = ", reg_rna_mse
      print "reg RNA expvar = ", reg_rna_expvar
      print "reg RNA corr   = ", reg_rna_corr
      
      #r2_score, mean_squared_error, explained_variance_score
      save_location_reg = os.path.join( logging_dict[SAVEDIR], "survival_predictions_reg.png" )  
      f2 = pp.figure()
      ax1 = f2.add_subplot(311)
      ax1.plot( reg_rna_class_projections[0], y_true_rna, 'b.' )
      #ax1.plot( dna_class_probabilties[0][I_dna_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "RNA predict" )
      pp.ylabel( "RNA target" )
      ax1 = f2.add_subplot(312)
      ax1.plot( reg_dna_class_projections[0], y_true_dna, 'b.' )
      #ax1.plot( dna_class_probabilties[0][I_dna_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "DNA predict" )
      pp.ylabel( "DNA target" )
      ax1 = f2.add_subplot(313)
      ax1.plot( reg_meth_class_projections[0], y_true_meth, 'b.' )
      #ax1.plot( dna_class_probabilties[0][I_dna_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "METH predict" )
      pp.ylabel( "METH target" )
      
      
      # ax2 = f2.add_subplot(1,4,2)
      # #ax1.plot( rna_class_y[I_rna_predictions], 'b.', alpha=0.5, mec='k' )
      # ax2.plot( rna_class_y[I_rna_predictions].cumsum()/float( rna_class_y.sum() ), 'b.', alpha=0.5, mec='k' )
      # ax2.plot( rna_class_probabilties[0][I_rna_predictions], 'r.', alpha=0.5, mec='k' )
      # pp.xlabel( "RNA roc=%0.2f acc=%0.2f"%(rna_auc, rna_acc) )
      # ax3 = f2.add_subplot(1,4,3)
      # #ax1.plot( meth_class_y[I_meth_predictions], 'b.', alpha=0.5, mec='k' )
      # ax3.plot( meth_class_y[I_meth_predictions].cumsum()/float( meth_class_y.sum() ), 'b.', alpha=0.5, mec='k' )
      # ax3.plot( meth_class_probabilties[0][I_meth_predictions], 'r.', alpha=0.5, mec='k' )
      # pp.xlabel( "METH roc=%0.2f acc=%0.2f"%(meth_auc, meth_acc) )
      # ax4 = f2.add_subplot(1,4,4)
      # #ax1.plot( meth_class_y[I_meth_predictions], 'b.', alpha=0.5, mec='k' )
      # ax4.plot( comb_class_y[I_comb_predictions].cumsum()/float( comb_class_y.sum() ), 'b.', alpha=0.5, mec='k' )
      # ax4.plot( comb_class_probabilties[0][I_comb_predictions], 'r.', alpha=0.5, mec='k' )
      # pp.xlabel( "COMB roc=%0.2f acc=%0.2f"%(comb_auc, comb_acc) )
      
      #tp,fn
      pp.savefig(save_location_reg, dpi=300, format='png')
      
      #pdb.set_trace()
      #print mn_prob
      I = np.argsort(-mn_prob)
      third = int(len(I)/3.0)
      half  = int(len(I)/2.0)
      I0 = I[:third]
      I1 = I[third:2*third]
      I2 = I[2*third:]
      
      group0 = I[:half]
      group1 = I[half:]
      
      
      dna_class_projections, \
      dna_class_probabilties, \
      dna_class_weights, \
      dna_class_averages, \
      dna_class_X, \
      dna_class_y = run_survival_prediction_loo( data_dict['validation_tissues'], f, d, group0, group1, data_keys = ["/DNA/channel/0"], data_names = ["DNA"], C = C )  
      
      #I_dna_prediction = np.argsort( dna_class_weights[0] )
      #print "DNA ranked high, ", [dna_class_X.columns[i] for i in I_dna_prediction[:10]] 
      #print "DNA ranked low, ", dna_class_X.columns[I_dna_prediction[-10:]]
      
      rna_class_projections, \
      rna_class_probabilties, \
      rna_class_weights, \
      rna_class_averages, \
      rna_class_X, \
      rna_class_y = run_survival_prediction_loo( data_dict['validation_tissues'], f, d, group0, group1, data_keys = ["/RNA/FAIR"], data_names = ["RNA"], C = C )  
      

      meth_class_projections, \
      meth_class_probabilties, \
      meth_class_weights, \
      meth_class_averages, \
      meth_class_X, \
      meth_class_y = run_survival_prediction_loo( data_dict['validation_tissues'], f, d, group0, group1, data_keys = ["/METH/FAIR"], data_names = ["METH"], C = C )  

      comb_class_projections, \
      comb_class_probabilties, \
      comb_class_weights, \
      comb_class_averages, \
      comb_class_X, \
      comb_class_y = run_survival_prediction_loo( data_dict['validation_tissues'], f, d, group0, group1, data_keys = ["/DNA/channel/0","/RNA/FAIR","/METH/FAIR"], data_names = ["DNA","RNA","METH"], C = C )  

      

      
      
      I_dna_parameters = np.argsort( -np.abs( dna_class_weights[0] ) )
      I_dna_predictions = np.argsort( dna_class_probabilties[0] )
      print "DNA ranked high, ", [dna_class_X.columns[i] for i in I_dna_parameters[:n2show]] 
      #print "DNA ranked low, ", [dna_class_X.columns[i] for i in I_dna_parameters[-n2show:]] 

      I_rna_parameters = np.argsort( -np.abs( rna_class_weights[0] ) )
      I_rna_predictions = np.argsort( rna_class_probabilties[0] )
      print "RNA ranked high, ", [rna_class_X.columns[i] for i in I_rna_parameters[:n2show]] 
      #print "RNA ranked low, ", [rna_class_X.columns[i] for i in I_rna_parameters[-n2show:]] 
      
      class_rna_Ws = rna_class_weights[2]
      
      save_location_reg = os.path.join( logging_dict[SAVEDIR], "survival_predictions_class_W.png" )  
      f0=pp.figure()
      ax1 = f0.add_subplot(121)
      ax1.plot(class_rna_Ws.T, 'k.', alpha=0.5)
      ax2 = f0.add_subplot(122)
      ax2.plot( class_rna_Ws.T.mean(1), reg_rna_Ws.T.mean(1), '.', alpha=0.5)
      pp.savefig(save_location_reg, dpi=300, format='png')
      
      I_meth_parameters = np.argsort( -np.abs( meth_class_weights[0] ) )
      I_meth_predictions = np.argsort( meth_class_probabilties[0] )
      print "METH ranked high, ", [meth_class_X.columns[i] for i in I_meth_parameters[:n2show]] 
      #print "METH ranked low, ", [meth_class_X.columns[i] for i in I_meth_parameters[-n2show:]] 

      I_comb_parameters = np.argsort( -np.abs( comb_class_weights[0] ) )
      I_comb_predictions = np.argsort( comb_class_probabilties[0] )
      print "COMB ranked high, ", [comb_class_X.columns[i] for i in I_comb_parameters[:n2show]] 
      #print "COMB ranked low, ", [comb_class_X.columns[i] for i in I_comb_parameters[-n2show:]] 

      pd.DataFrame( dna_class_weights[0][I_dna_parameters], columns=["%s_DNA_class"%disease],index = [dna_class_X.columns[i] for i in I_dna_parameters] ).to_csv(save_weights_template+"dna_class.csv")
      pd.DataFrame( rna_class_weights[0][I_rna_parameters], columns=["%s_RNA_class"%disease],index = [rna_class_X.columns[i] for i in I_rna_parameters] ).to_csv(save_weights_template+"rna_class.csv")
      pd.DataFrame( meth_class_weights[0][I_meth_parameters], columns=["%s_METH_class"%disease],index = [meth_class_X.columns[i] for i in I_meth_parameters] ).to_csv(save_weights_template+"meth_class.csv")
      pd.DataFrame( comb_class_weights[0][I_comb_parameters], columns=["%s_COMBO_class"%disease],index = [comb_class_X.columns[i] for i in I_comb_parameters] ).to_csv(save_weights_template+"comb_class.csv")

      pd.DataFrame( dna_class_weights[0][I_dna_parameters], columns=["%s_DNA_class"%disease],index = [dna_class_X.columns[i] for i in I_dna_parameters] ).to_csv(save_weights_template+"dna_class.txt", sep=" ")
      pd.DataFrame( rna_class_weights[0][I_rna_parameters], columns=["%s_RNA_class"%disease],index = [rna_class_X.columns[i] for i in I_rna_parameters] ).to_csv(save_weights_template+"rna_class.txt", sep=" ")
      pd.DataFrame( meth_class_weights[0][I_meth_parameters], columns=["%s_METH_class"%disease],index = [meth_class_X.columns[i] for i in I_meth_parameters] ).to_csv(save_weights_template+"meth_class.txt", sep=" ")
      pd.DataFrame( comb_class_weights[0][I_comb_parameters], columns=["%s_COMBO_class"%disease],index = [comb_class_X.columns[i] for i in I_comb_parameters] ).to_csv(save_weights_template+"comb_class.txt", sep=" ")
      
      
      dna_auc = roc_auc_score( dna_class_y, dna_class_probabilties[0] )
      rna_auc = roc_auc_score( rna_class_y, rna_class_probabilties[0] )
      meth_auc = roc_auc_score( meth_class_y, meth_class_probabilties[0] )
      comb_auc = roc_auc_score( comb_class_y, comb_class_probabilties[0] )
      
      d_fpr, d_tpr, d_thresholds = roc_curve( np.squeeze(dna_class_y), np.squeeze(dna_class_probabilties[0]), pos_label=2)
      r_fpr, r_tpr, r_thresholds = roc_curve(rna_class_y, rna_class_probabilties[0], pos_label=2)
      m_fpr, m_tpr, m_thresholds = roc_curve(meth_class_y, meth_class_probabilties[0], pos_label=2)
      c_fpr, c_tpr, c_thresholds = roc_curve(comb_class_y, comb_class_probabilties[0], pos_label=2)
      
      dna_acc  = accuracy_score(dna_class_y, dna_class_probabilties[0]>0.5) 
      rna_acc  = accuracy_score(rna_class_y, rna_class_probabilties[0]>0.5) 
      meth_acc = accuracy_score(meth_class_y, meth_class_probabilties[0]>0.5) 
      comb_acc = accuracy_score(comb_class_y, comb_class_probabilties[0]>0.5) 
      
      rna_reg_auc   = roc_auc_score( rna_class_y, y_pred_rna )
      dna_reg_auc   = roc_auc_score( dna_class_y, y_pred_dna )
      meth_reg_auc  = roc_auc_score( meth_class_y, y_pred_meth )
      combo_reg_auc = roc_auc_score( comb_class_y, y_pred_combo )
      
      print "DNA_ROC  = ", dna_auc
      print "DNA_ROC (REG)  = ", dna_reg_auc
      print "RNA_ROC  = ", rna_auc
      print "RNA_ROC (REG)  = ", rna_reg_auc
      print "METH_ROC = ", meth_auc
      print "METH_ROC (REG)  = ", meth_reg_auc
      print "COMB_ROC = ", comb_auc
      print "COMB_ROC (REG)  = ", combo_reg_auc
      
      print "DNA_ACC  = ", dna_acc
      print "RNA_ACC  = ", rna_acc
      print "METH_ACC = ", meth_acc
      print "COMB_ACC = ", comb_acc
      
      save_location_pred = os.path.join( logging_dict[SAVEDIR], "survival_predictions.png" )  
      f2 = pp.figure()
      ax1 = f2.add_subplot(1,4,1)
      ax1.plot( dna_class_y[I_dna_predictions].cumsum()/float( dna_class_y.sum() ), 'b.', alpha=0.5, mec='k' )
      #ax1.plot( dna_class_y[I_dna_predictions], 'b.', alpha=0.5, mec='k' )
      ax1.plot( dna_class_probabilties[0][I_dna_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "DNA roc=%0.2f acc=%0.2f"%(dna_auc, dna_acc) )
      ax2 = f2.add_subplot(1,4,2)
      #ax1.plot( rna_class_y[I_rna_predictions], 'b.', alpha=0.5, mec='k' )
      ax2.plot( rna_class_y[I_rna_predictions].cumsum()/float( rna_class_y.sum() ), 'b.', alpha=0.5, mec='k' )
      ax2.plot( rna_class_probabilties[0][I_rna_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "RNA roc=%0.2f acc=%0.2f"%(rna_auc, rna_acc) )
      ax3 = f2.add_subplot(1,4,3)
      #ax1.plot( meth_class_y[I_meth_predictions], 'b.', alpha=0.5, mec='k' )
      ax3.plot( meth_class_y[I_meth_predictions].cumsum()/float( meth_class_y.sum() ), 'b.', alpha=0.5, mec='k' )
      ax3.plot( meth_class_probabilties[0][I_meth_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "METH roc=%0.2f acc=%0.2f"%(meth_auc, meth_acc) )
      ax4 = f2.add_subplot(1,4,4)
      #ax1.plot( meth_class_y[I_meth_predictions], 'b.', alpha=0.5, mec='k' )
      ax4.plot( comb_class_y[I_comb_predictions].cumsum()/float( comb_class_y.sum() ), 'b.', alpha=0.5, mec='k' )
      ax4.plot( comb_class_probabilties[0][I_comb_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "COMB roc=%0.2f acc=%0.2f"%(comb_auc, comb_acc) )
      
      #tp,fn
      pp.savefig(save_location_pred, dpi=300, format='png') 

      save_location_roc = os.path.join( logging_dict[SAVEDIR], "survival_predictions_roc.png" )  
      f2 = pp.figure()
      ax1 = f2.add_subplot(1,4,1)
      ax1.plot( dna_class_y[I_dna_predictions], 'b.' )
      ax1.plot( dna_class_probabilties[0][I_dna_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "DNA roc=%0.2f acc=%0.2f"%(dna_auc, dna_acc) )
      ax2 = f2.add_subplot(1,4,2)
      ax2.plot( rna_class_y[I_rna_predictions], 'b.', alpha=0.5, mec='k' )
      ax2.plot( rna_class_probabilties[0][I_rna_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "RNA roc=%0.2f acc=%0.2f"%(rna_auc, rna_acc) )
      ax3 = f2.add_subplot(1,4,3)
      ax3.plot( meth_class_y[I_meth_predictions], 'b.', alpha=0.5, mec='k' )
      ax3.plot( meth_class_probabilties[0][I_meth_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "METH roc=%0.2f acc=%0.2f"%(meth_auc, meth_acc) )      
      ax4 = f2.add_subplot(1,4,4)
      ax4.plot( comb_class_y[I_comb_predictions], 'b.', alpha=0.5, mec='k' )
      ax4.plot( comb_class_probabilties[0][I_comb_predictions], 'r.', alpha=0.5, mec='k' )
      pp.xlabel( "METH roc=%0.2f acc=%0.2f"%(comb_auc, comb_acc) )      
      #tp,fn
      pp.savefig(save_location_roc, dpi=300, format='png') 

      
      #pdb.set_trace()
      kmf = KaplanMeierFitter()
      if len(I2) > 0:
        kmf.fit(T_train[I2], event_observed=E_train[I2], label =  "lda_1 E=%d C=%d"%(E_train[I2].sum(),len(I2)-E_train[I2].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
      if len(I1) > 0:
        kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
        ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='green')
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
    weights_matrix.append( weights[0] )
    s.close()
    d.close()
    f.close()  
  
  
######################################################################################################
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file  
  weights_matrix = []
  main( yaml_file, weights_matrix )

  
  
  