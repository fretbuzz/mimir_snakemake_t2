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

  # TODO: mod this.
  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_60_exp.json | tee sockshop_four_60.log;\

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_160_exp.json | tee sockshop_four_160.log;\

  sleep 60

  sudo minikube stop || true
fi

  # TODO: call splitter function here (first I'll need to write it...)

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh | tee install_mimr_depend_log.txt

mkdir multilooper_outs

# TODO: mod this
sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json | tee ./multilooper_outs/new_sockshop_scale.log

# sudo python generate_paper_graphs.py --dont_update_config --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json

ls
####