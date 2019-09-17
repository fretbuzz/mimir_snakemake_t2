#!/usr/bin/env bash
### SOCKOSHOP REPRODUCABILITY -- SCALE -- WITH PHYSICAL ATTACKS!!!! ###

# this should be applicable to everyone...
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh

# need to install DET locally as part of physical exfil component
sudo git clone https://github.com/fretbuzz/DET.git /DET/

sudo python run_experiment.py --prepare_app_p --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale_Physical/sockshop_four_100_exp.json ;\

sudo python run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale_Physical/sockshop_four_100_mk2_exp.json ;\

sleep 60

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

sudo python generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale_PHYSICAL/sockshop_scale_physical.json ;\

exit 1





echo "add in the rest of this later-- depending on how we want to do it..."

sudo python run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_mk2_exp.json ;\
sudo python run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_120_exp.json;\
sudo python run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_140_exp.json

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python run_experiment.py --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_160_exp.json;\
sudo python run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_40_exp.json ;\
sudo python run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_60_exp.json ;\
sudo python run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_80_exp.json

sleep 60

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

sudo python generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json

# sudo python generate_paper_graphs.py --dont_update_config --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json

ls
####