

class PoissonRsem(object):
  def __init__( self, arch_dict, data_dict ):
    self.arch_dict = arch_dict
    self.data_dict = data_dict
    
  def train(self, algo_dict, logging_dict ):
    self.algo_dict = algo_dict 
    self.logging_dict = logging_dict
    