from tcga_encoder.utils.helpers import *
from scipy import stats

def auc_standard_error( theta, nA, nN ):
  # from: Hanley and McNeil (1982), The Meaning and Use of the Area under the ROC Curve
  # theta: estimated AUC, can be 0.5 for a random test
  # nA size of population A
  # nN size of population N
  
  Q1=theta/(2.0-theta); Q2=2*theta*theta/(1+theta)
  
  SE = np.sqrt( (theta*(1-theta)+(nA-1)*(Q1-theta*theta) + (nN-1)*(Q2-theta*theta) )/(nA*nN) )
  
  return SE
  
def auc_p_value( auc1, auc2, std_error1, std_error2 ):
  
  se_combined = np.sqrt( std_error1**2 + std_error2**2 )
  
  difference = auc1 - auc2
  z_values = difference / se_combined 
  sign_difference = np.sign(difference)
  
  p_values = 1.0 - stats.norm.cdf( np.abs(z_values) ) 
  
  return p_values
  
  