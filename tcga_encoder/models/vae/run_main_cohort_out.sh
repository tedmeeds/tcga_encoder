#!/bin/bash
yaml_template_file="$1"
#declare -i n_xval_folds
#n_xval_folds="$1"
declare -i run_it
run_it="$2"
main_script="tcga_encoder/models/vae/main_cohort_out.py"
cohort_groups="tcga_encoder/data/disease_groups.txt"
#collect_script="tcga_encoder/models/dna/main_cohort_out_collector_simple.py"

echo "yaml script: $yaml_template_file"
echo "cohorts: $cohort_groups"
while IFS= read line
do
        # display $line or do somthing with $line
	echo "$line"
      s="$main_script $yaml_template_file $line"
      echo "+++++++++++++++++++++++++++++++++++++++++"
      echo "+++++++++++++++++++++++++++++++++++++++++"
      echo "+++++++++++++++++++++++++++++++++++++++++"
      echo "+++++++++++++++++++++++++++++++++++++++++"
      echo $s
      echo "+++++++++++++++++++++++++++++++++++++++++"
      echo "+++++++++++++++++++++++++++++++++++++++++"
      echo "+++++++++++++++++++++++++++++++++++++++++"
      if [ "$run_it" -eq 1 ]; then
        python $s
      fi
done <"$cohort_groups"


