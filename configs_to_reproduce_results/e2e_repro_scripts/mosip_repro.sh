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
  # vv I think not needed... MOSIP launcer will do it all for us...
  #bash ../configs_to_reproduce_results/kubernetes_setup_script_centos.sh | tee kubernetes_setup.log;
  sudo pip uninstall pyOpenSSL
  # TODO: honestly, using MOSIP launcher is probably easier...
  ### yes, we'll need an LDAP server... try following this tutorial: https://linuxhostsupport.com/blog/how-to-install-ldap-on-centos-7/
  '''
  sudo yum install https://releases.ansible.com/ansible/rpm/release/epel-7-x86_64/ansible-2.7.9-1.el7.ans.noarch.rpm
  sudo easy_install pip
  sudo pip install jmespath==0.9.4
  '''
  git clone https://github.com/mosip/mosip-platform.git
  git clone https://github.com/mosip/mosip-ref-impl.git
  git clone https://github.com/mosip/mosip-config.git
  git clone https://github.com/mosip/mosip-infra.git
  ################# probably do NOT do this... probably too complicated... just use MOSIP launcher...
  '''
  sudo yum install -y maven
  cd ./mosip-platform
  #mvn clean install
  # TODO: should automate this (i did it manually...)
    # (i) remove authentication/authentication-demo-app
    # (2) remove bioSDK part from registration/registration-services/pom.xml
  mvn clean install -Dmaven.test.skip=true

  # TODO: add code to setup private docker registry and build MOSIP components...
  # deploy registry docker
  run -d -p 5000:5000 --restart always --name registry registry:2
  # then push from the maven artificat repository to the docker registry

  # TODO push the moven images to the docker registry
  # docker image list (hopefully it shows up... if it does then we push to the registry...)
  # ^^ just to check during development, not part of long-term workflow...

  # then modify the kubernetes config files to grab from our docker registery
  # TODO

  # finaly, execute kubectl apply to those files...
  # TODO
  '''
  ###################

  # https://get.helm.sh/helm-v3.0.0-linux-amd64.tar.gz
  # tar -zxvf helm-v3.0.0-linux-amd64.tar.gz
  # /usr/local/bin/helm


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
