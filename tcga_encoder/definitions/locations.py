import os

LOCATION          = "location"
HOME_DIR          =  os.environ["HOME"]
RESULTS_DIR       = os.path.join( os.environ["HOME"], "results" )
PRETRAINED_DIR    = os.path.join( RESULTS_DIR, "pretrained")
EXPERIMENTS_DIR   = os.path.join( RESULTS_DIR, "experiments")
EXPERIMENT        = "experiment"
SAVEDIR           = "savedir"
LOGGING           = "logging"
