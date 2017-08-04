from tcga_encoder.utils.helpers import *
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

class PanCancerSurvival(object):
  def __init__( self, data_store ):
    ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death","patient.days_to_birth"]]
    tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
    surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
    NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns )
    NEW_SURVIVAL = NEW_SURVIVAL.loc[surv_barcodes]
    #clinical = data_store["/CLINICAL/data"].loc[barcodes]

    Age = NEW_SURVIVAL[ "patient.days_to_birth" ].values.astype(int)
    Times = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+NEW_SURVIVAL[ "patient.days_to_death" ].fillna(0).values.astype(int)
    Events = (1-np.isnan( NEW_SURVIVAL[ "patient.days_to_death" ].astype(float)) ).astype(int)

    ok_age_query = Age<-10
    ok_age = pp.find(ok_age_query )
    #tissues = tissues[ ok_age_query ]
    #pdb.set_trace()
    Age=-Age[ok_age]
    Times = Times[ok_age]
    Events = Events[ok_age]
    s_barcodes = surv_barcodes[ok_age]
    NEW_SURVIVAL = NEW_SURVIVAL.loc[s_barcodes]

    bad_followup_query = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)<0
    bad_followup = pp.find( bad_followup_query )

    ok_followup_query = 1-bad_followup_query
    ok_followup = pp.find( ok_followup_query )

    bad_death_query = NEW_SURVIVAL[ "patient.days_to_death" ].fillna(0).values.astype(int)<0
    bad_death = pp.find( bad_death_query )

    #pdb.set_trace()
    Age=Age[ok_followup]
    Times = Times[ok_followup]
    Events = Events[ok_followup]
    s_barcodes = s_barcodes[ok_followup]
    NEW_SURVIVAL = NEW_SURVIVAL.loc[s_barcodes]
  
    S = pd.DataFrame( np.vstack((Events,Times)).T, index = s_barcodes, columns=["E","T"])
    
    self.Events = Events
    self.Times  = Times
    self.barcodes_ok = s_barcodes 
    self.Age = Age
    self.data = S
    self.bad_death = bad_death
    self.ok_followup = ok_followup
    self.bad_followup = bad_followup
    
def groups_by_splits( n, split_indices ):
  groups = np.zeros(n)
  k = 1
  for splits in split_indices[1:]:
    groups[splits] = k; k+=1
  return groups
  
  
def  survival_splits( events, event_order, split_nbr ):
  split_indices = []
  
  cum_events = events[ event_order ].cumsum()
  
  n_events = events.sum()
  event_fraction = float( n_events ) / float( split_nbr )
  min_nbr = -1.0
  for k in range( split_nbr ):
    at_least = cum_events > min_nbr
    at_most  = cum_events <= (k+1)*event_fraction
    
    ids = pp.find( at_least & at_most )
    min_nbr = (k+1)*event_fraction
    
    split_indices.append( event_order[ ids ] )
  
  return split_indices


def plot_survival_by_splits( times, events, split_indices, at_risk_counts=False,show_censors=True,ci_show=False, cmap = "rainbow", colors = None, labels=None):
  
  split_nbr = len(split_indices)
  
  f = pp.figure()
  ax= f.add_subplot(111)
  kmf = KaplanMeierFitter()
  if colors is None:
    c = pp.get_cmap(cmap)
    colors = c( np.linspace(0,256,len(split_indices)).astype(int) )
  
  k=0
  for splits in split_indices:
    #rgb_color = c( int( c.N*float(k) / (len(split_indices)+1 ) ) )
    #print rgb_color, k, c.N*float(k) / (len(split_indices)+1 )
    #pdb.set_trace()
    if labels is None:
      kmf.fit(times[splits], event_observed=events[splits], label="q=%d/%d"%(k+1,split_nbr)  )
    else:
      kmf.fit(times[splits], event_observed=events[splits], label=labels[k]  )
      
    ax=kmf.plot(ax=ax,at_risk_counts=at_risk_counts,show_censors=show_censors,ci_show=ci_show, color=colors[k],lw=3)
    k+=1
  pp.ylim(0,1)
  
  return ax
        
        