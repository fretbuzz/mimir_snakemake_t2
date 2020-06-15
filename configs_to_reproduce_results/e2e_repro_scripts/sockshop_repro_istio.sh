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

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_60_exp_small_scale.json | tee sockshop_four_60_exp_small_scale.log;\