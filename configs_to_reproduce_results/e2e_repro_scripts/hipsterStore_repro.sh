#!/usr/bin/env bash
### HipsterStore REPRODUCABILITY -- SCALE ###
# NOTE: adservice doesn't work-- but need to wait for the dev's to fix it...

# this should be applicable to everyone...
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh

sudo python run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_100_mk2_exp.json ;
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_100_exp.json ;
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_120_exp.json ;
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_140_exp.json ;

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_160_exp.json ;
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_80_exp.json ;
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_60_exp.json ;
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_40_exp.json ;

sleep 60

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

sudo python generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/HipsterStore/Scale/hipsterStore_scale.json


echo "okieee, all done :)"