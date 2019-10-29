#!/usr/bin/env bash
### SOCKOSHOP REPRODUCABILITY -- SCALE ###
## THIS SHOULD be all that I need to do to get the Sockshop reproducibility thing going ##
## TODO: update for the new directory

if [ $# -gt 2 ]; then
  echo "too many args";
  exit 2;
fi

skip_pcap=0
if [ "$1" == "--skip_pcap" ]; then
  skip_pcap=1
fi

echo "skip_pcap $skip_pcap"

if [ $skip_pcap -eq 0 ]
then

  # this should be applicable to everyone...
  bash ../configs_to_reproduce_results/kubernetes_setup_script.sh | tee kubernetes_setup.log;

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_160_exp.json | tee sockshop_four_160.log;\
  sleep 120
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_mk2_exp.json | tee sockshop_four_100_mk2.log;\
  sleep 120
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_120_exp.json | tee sockshop_four_120.log;\

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_140_exp.json | tee sockshop_four_140.log
  sleep 120
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_exp.json | tee sockshop_four_100.log;\

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_40_exp.json | tee sockshop_four_40.log;\
  sleep 120
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_60_exp.json | tee sockshop_four_60.log;\
  sleep 120
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_80_exp.json | tee sockshop_four_80.log

  sleep 60

  sudo minikube stop || true

fi

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json | tee new_sockshop_scale.log

# sudo python generate_paper_graphs.py --dont_update_config --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json

ls
####