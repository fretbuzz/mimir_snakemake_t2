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
  # TODO: replace the vars (in ansible var file)...
  # use this as a basis: sed -i ?? mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/mosip.kernel.sms.gateway=<ToBeReplaced>/mosip.kernel.sms.gateway=nonsenseValue/" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/mosip.kernel.sms.api=<ToBeReplaced>/mosip.kernel.sms.api=nonsenseValue/" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/mosip.kernel.sms.username=<ToBeReplaced>/mosip.kernel.sms.username=nonsenseValue/" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/mosip.kernel.sms.password=<ToBeReplaced>/mosip.kernel.sms.password=nonsenseValue/" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/mosip.kernel.sms.sender=<ToBeReplaced>/mosip.kernel.sms.sender=nonsenseValue/" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties

  hostIp=$(curl ifconfig.me)
  # TODO: also replace the values related to the mail server... try using a fake mail server, like here:
  # https://serverfault.com/questions/207619/how-to-setup-a-fake-smtp-server-to-catch-all-mails

  sudo python -m smtpd -n -c DebuggingServer localhost:25 &
  sed -i "s/spring.mail.username=<ToBeReplaced>/spring.mail.username=nonsenseValue/" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/spring.mail.password=<ToBeReplaced>/spring.mail.password=nonsenseValue/" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/spring.mail.host=<ToBeReplaced>/spring.mail.host=127.0.0.1" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties
  sed -i "s/spring.mail.port=<ToBeReplaced>/spring.mail.port=25" mosip-infra/deployment/sandbox/playbooks-properties/all-playbooks.properties

  cd mosip-infra/deployment/sandbox/
  sudo sh install-mosip-kernel.sh
  sleep 360
  echo "about to start running pre-registration module..."
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
