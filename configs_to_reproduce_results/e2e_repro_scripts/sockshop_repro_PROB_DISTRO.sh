#!/usr/bin/env bash

# this should be applicable to everyone...
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh | tee kubernetes_setup.log;

sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_5_3_2_exp_mk2.json | tee sockshop_four_5_3_2_exp_mk2.log;

sleep 60

sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_0_0_1_exp.json | tee sockshop_four_0_0_1_exp.log;

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_0_1_0_exp.json | tee sockshop_four_0_1_0_exp.log;

sleep 60

sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_0_5_5_exp.json | tee sockshop_four_0_5_5_exp.log;

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_1_0_0_exp.json | tee sockshop_four_1_0_0_exp.log;

sleep 60

sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_4_35_25.json | tee sockshop_four_4_35_25.log;

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_5_0_5_exp.json | tee sockshop_four_5_0_5_exp.log;

sleep 60

sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_55_2_25_exp.json | tee sockshop_four_55_2_25_exp.log;

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_55_35_1_exp.json | tee sockshop_four_55_35_1_exp.log;

sleep 60

sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Prob_Distro/sockshop_four_5_3_2_exp.json | tee sockshop_four_5_3_2_exp.log;

sleep 60

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Prob_Distro/new_sockshop_angle.json | tee new_sockshop_angle.log;
# --dont_update_config

ls
####