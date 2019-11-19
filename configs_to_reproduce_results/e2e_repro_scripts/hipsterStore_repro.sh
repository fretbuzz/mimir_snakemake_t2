#!/usr/bin/env bash
### HipsterStore REPRODUCABILITY -- SCALE ###
# NOTE: adservice doesn't work-- but need to wait for the dev's to fix it...

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

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_100_mk2_exp.json | tee hipsterStore_100_mk2.log;
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_100_exp.json | tee hipsterStore_100.log;
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_120_exp.json | tee hipsterStore_120.log;
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_140_exp.json | tee hipsterStore_140.log;

  sleep 60

  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_160_exp.json | tee hipsterStore_160.log;
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_80_exp.json | tee hipsterStore_80.log;
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_60_exp.json | tee hipsterStore_60.log;
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/HipsterStore/Scale/hipsterStore_40_exp.json | tee hipsterStore_40.log;

  sleep 60

  sudo minikube stop || true

fi

cd ../analysis_pipeline/

## then install the analysis pipeline depdencies
. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

mkdir multilooper_outs

sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/HipsterStore/Scale/hipsterStore_scale.json | tee tee ./multilooper_outs/hipsterStore_repro.log;


echo "okieee, all done :)"


