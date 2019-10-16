#!/usr/bin/env bash
### SOCKOSHOP REPRODUCABILITY -- SCALE -- WITH PHYSICAL ATTACKS!!!! ###

# this should be applicable to everyone...
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh | tee kubernetes_setup.log;

cd ../analysis_pipeline/dlp_stuff/

. ../../configs_to_reproduce_results/install_decanter_script.sh

cd ../../experiment_coordinator

# need to install DET locally as part of physical exfil component
sudo git clone https://github.com/fretbuzz/DET.git /DET/

sudo python -u run_experiment.py --prepare_app_p --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale_Physical/sockshop_four_100_exp.json | tee sockshop_four_100.log;\

sudo python -u run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale_Physical/sockshop_four_100_mk2_exp.json | tee sockshop_four_100_mk2.log;\

sleep 60

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale_PHYSICAL/sockshop_scale_physical.json | tee sockshop_scale_physical.log;\

exit 1





echo "add in the rest of this later-- depending on how we want to do it..."

sudo python -u run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_mk2_exp.json ;\
sudo python -u run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_120_exp.json;\
sudo python -u run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_140_exp.json

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python -u run_experiment.py --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_160_exp.json;\
sudo python -u run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_40_exp.json ;\
sudo python -u run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_60_exp.json ;\
sudo python -u run_experiment.py --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_80_exp.json

sleep 60

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json

# sudo python generate_paper_graphs.py --dont_update_config --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json

ls
####