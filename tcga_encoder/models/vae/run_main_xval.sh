#!/bin/bash
yaml_template_file="$1"
declare -i n_xval_folds
n_xval_folds="$2"
declare -i run_it
run_it="$3"
main_script="tcga_encoder/models/vae/main_xval.py"
collect_script="tcga_encoder/models/dna/main_xval_collector_simple.py"

echo "yaml script: $yaml_template_file"
echo "nbr folds: $n_xval_folds"
#echo "train? $run_it"
echo {1..$n_xval_folds}
if [ "$run_it" -eq 1 ]; then
  for (( fold=1; fold<=$n_xval_folds; fold++ )) 
  do
    #echo "$main_script" "$yaml_template_file" "$fold" "$n_xval_folds"
    s="$main_script $yaml_template_file $fold $run_it"
    echo $s
    python $s
  done
fi

echo "$collect_script" "$yaml_template_file" "$n_xval_folds"
python "$collect_script" "$yaml_template_file" "$n_xval_folds"

