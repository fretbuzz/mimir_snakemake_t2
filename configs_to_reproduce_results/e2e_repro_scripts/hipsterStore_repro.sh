#!/usr/bin/env bash
### HipsterStore REPRODUCABILITY -- SCALE ###

# this should be applicable to everyone...
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh

sudo python run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_100_mk2_exp.json ;

echo "okieee, all done :)"