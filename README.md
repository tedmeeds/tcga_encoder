## tcga_encoder
Encoding Heterogeneous Molecular Cancer Profiles Using Deep Neural Networks


# /data
Scripts for preparing HDF files for training models.  Adjust these to select genes for different sources, such as MUTSIG for DNA mutations, etc.

# /experiments
YAML files for defining models, data, and algorithms for training models.  Also defines experiment names.

# /models
Tensorflow wrappers that define specific types models by directory, eg /models/vae.  For example, VanillaVariationalAutoEncoder.  Also location of algorithm, layer, regularizer wrappers.  Mains are found in the model type directories.

# /utils
Helper functions.

# /viz
Various visualisation scripts.