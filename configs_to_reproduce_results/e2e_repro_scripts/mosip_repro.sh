#!/usr/bin/env bash
### SOCKOSHOP REPRODUCABILITY -- SCALE ###
## THIS SHOULD be all that I need to do to get the Sockshop reproducibility thing going ##
## TODO: update for the new directory

apt-get update

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

  # use this method for deployment https://github.com/mosip/mosip-infra/tree/master/deployment/sandbox
  # note: this is all untested and will certainly fail...
  git clone https://github.com/mosip/mosip-infra || true
  # TODO: replace the vars...
  cd mosip-infra/deployment/sandbox/
  sudo sh install-mosip-kernel.sh
  sleep 360
  sudo sh install-mosip-pre-reg.sh
  seep 720

  # TODO: run the MOSIP experiments here...

  sudo minikube stop || true

else
  # note: this part looks like the other experiments b/c we don't run MIMIR on CentOS...
  # (so for MOSIP experiment you need 2 machines... one CentOS to get the PCAPS and an Ubuntu 14.04 to analyze them)

  cd ../analysis_pipeline/

  ## then install the analysis pipeline depdencies
  . ../configs_to_reproduce_results/install_mimir_depend_scripts.sh | tee install_mimr_depend_log.txt

  mkdir multilooper_outs

  # TODO: run analysis here...

  ls
  ####
fi
