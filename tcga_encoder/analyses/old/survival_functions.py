from tcga_encoder.utils.helpers import *
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

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


def plot_survival_by_splits( times, events, split_indices, at_risk_counts=False,show_censors=True,ci_show=False):
  
  split_nbr = len(split_indices)
  
  f = pp.figure()
  ax= f.add_subplot(111)
  kmf = KaplanMeierFitter()
  k=0
  for splits in split_indices:
    kmf.fit(times[splits], event_observed=events[splits], label="q=%d/%d"%(k+1,split_nbr)  )
    ax=kmf.plot(ax=ax,at_risk_counts=at_risk_counts,show_censors=show_censors,ci_show=ci_show)
    k+=1
  pp.ylim(0,1)
  
  return ax
        
        