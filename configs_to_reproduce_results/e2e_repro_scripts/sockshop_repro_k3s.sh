#!/usr/bin/env bash
### SOCKOSHOP REPRODUCABILITY -- SCALE ###
## THIS SHOULD be all that I need to do to get the Sockshop reproducibility thing going ##
## TODO: update for the new directory

if [ $# -gt 3 ]; then
  echo "too many args";
  exit 2;
fi

skip_pcap=0
if [ "$1" == "--skip_pcap" ]; then
  skip_pcap=1
fi

echo "skip_pcap $skip_pcap"

use_k3s_cluster=0
if [ "$1" == "--use_k3s_cluster" ]; then
  use_k3s_cluster=1
fi

echo "use_k3s_cluster $use_k3s_cluster"

if [ $skip_pcap -eq 0 ]
then

  # this should be applicable to everyone...
  if [ $use_k3s_cluster -eq 0 ]
  then
    echo "bash ../configs_to_reproduce_results/kubernetes_setup_script.sh"
    bash ../configs_to_reproduce_results/kubernetes_setup_script.sh | tee kubernetes_setup.log;
  else
    echo "bash ../configs_to_reproduce_results/kubernetes_setup_script.sh --use_k3s_cluster"
    . ../configs_to_reproduce_results/kubernetes_setup_script.sh --use_k3s_cluster | tee kubernetes_setup.log;
    print "okay, k3s should be setup... about to perform run_experiment..."
  fi

  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_60_exp.json | tee sockshop_four_60.log;\

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_160_exp.json | tee sockshop_four_160.log;\
  sleep 120
  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_mk2_exp.json | tee sockshop_four_100_mk2.log;\
  sleep 120
  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_120_exp.json | tee sockshop_four_120.log;\

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_140_exp.json | tee sockshop_four_140.log
  sleep 120
  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_exp.json | tee sockshop_four_100.log;\

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_40_exp.json | tee sockshop_four_40.log;\

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --use_k3s_cluster --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_80_exp.json | tee sockshop_four_80.log

  sleep 60

  sudo minikube stop || true

fi

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh | tee install_mimr_depend_log.txt

mkdir multilooper_outs

sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json | tee ./multilooper_outs/new_sock_all_exfilRate_sameTime.log

# will these next commands work?? probably not without a significant amount of debugging...
# note: i don't think these are actually needed for any reason...
#sudo python -u generate_paper_graphs.py --load_old_pipelines --retrain_model --min_exfil_rate_model --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json | tee ./multilooper_outs/new_sock_minRate_tsl.log

#sudo python -u generate_paper_graphs.py --load_old_pipelines --retrain_model --per_svc_exfil_model --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json | tee ./multilooper_outs/new_sock_perSvc.log

ls
####