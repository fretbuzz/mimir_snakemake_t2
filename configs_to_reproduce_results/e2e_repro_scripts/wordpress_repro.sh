### WORDPRESS REPRODUCABILITY -- SCALE ###
## THIS SHOULD be all that I need to do to get the wordpress reproducibility thing going ##

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
  # should be applicable to everyone
  bash ../configs_to_reproduce_results/kubernetes_setup_script.sh | tee kubernetes_setup.log;

  # then do wordpress-specific setup
  sudo bash install_scripts/install_selenium_dependencies.sh | tee wordpress_setup_log.txt;
  source ~/.bashrc # try this if it crashes again...
  bash ../configs_to_reproduce_results/setup_wordpress.sh | tee -a wordpress_setup_log.txt;
  MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
  host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
  WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"

  # then run some expreiments
  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_85.json | tee wordpress_85.log

  ## then cycle minikube/wordpress down/up
  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  bash ../configs_to_reproduce_results/setup_wordpress.sh
  MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
  host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
  WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"

  sleep 120

  # time for next batch of experiments
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_85_mk2.json | tee wordpress_85_mk2.log

  ## then cycle minikube/wordpress down/up
  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  bash ../configs_to_reproduce_results/setup_wordpress.sh
  MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
  host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
  WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"

  sleep 120

  # time for next batch of experiments

  sudo python -u run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_65.json | tee wordpress_65.log

  ## then cycle minikube/wordpress down/up
  bash ../configs_to_reproduce_results/cycle_minikube.sh

  sleep 120

  bash ../configs_to_reproduce_results/setup_wordpress.sh
  MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
  host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
  WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"

  sleep 120

  # time for next batch of experiments
  sudo python -u run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_45.json | tee wordpress_45.log

  sleep 60

  sudo minikube stop || true
fi

## cycle it there.
bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

#sudo python wordpress_setup/scale_wordpress.py 7
#MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
#host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
#WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"
#cd ./wordpress_setup/
#python setup_wordpress.py $MINIKUBE_IP $WORDPRESS_PORT "hi"
#cd ..
bash ../configs_to_reproduce_results/setup_wordpress.sh
MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"

sleep 120

sudo python -u run_experiment.py --no_exfil --prepare_app  --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_105.json | tee wordpress_105.log

## then cycle minikube/wordpress down/up
bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

bash ../configs_to_reproduce_results/setup_wordpress.sh
MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"

sleep 120

# time for next batch of experiments
sudo python -u run_experiment.py --no_exfil --prepare_app  --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_125.json | tee wordpress_125.log

sleep 120

cd ../analysis_pipeline/

. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

mkdir multilooper_outs

sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Wordpress/Scale/wordpress_scale.json | tee ./multilooper_outs/wordpress_scale.log

ls