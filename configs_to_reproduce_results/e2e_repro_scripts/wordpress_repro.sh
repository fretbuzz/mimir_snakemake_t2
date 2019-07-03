### WORDPRESS REPRODUCABILITY -- SCALE ###
## THIS SHOULD be all that I need to do to get the wordpress reproducibility thing going ##

# should be applicable to everyone
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh

# then do wordpress-specific setup
bash install_scripts/install_selenium_dependencies.sh
source ~/.bashrc # try this if it crashes again...
bash ../configs_to_reproduce_results/setup_wordpress.sh
MINIKUBE_IP=$(sudo minikube ip) # this theoretically ends whatever script is being used
host=$(sudo minikube service wwwppp-wordpress --url | tail -n 1)
WORDPRESS_PORT="$(echo $host | sed -e 's,^.*:,:,g' -e 's,.*:\([0-9]*\).*,\1,g' -e 's,[^0-9],,g')"
# this part is done in the ../configs_to_reproduce_results/setup_wordpress.sh script
#cd ./wordpress_setup/
#python setup_wordpress.py $MINIKUBE_IP $WORDPRESS_PORT "hi"
#cd ..

# then run some expreiments
sudo python run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_85.json

sleep 180

sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_85_mk2.json

## then cycle minikube/wordpress down/up
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

# time for next batch of experiments

sudo python run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_65.json

sleep 180

sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_45.json

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

sudo python run_experiment.py --no_exfil --prepare_app  --config_file ../configs_to_reproduce_results/Data_Collection/Wordpress/Scale/wordpress_105.json

sleep 120

cd ../analysis_pipeline/

. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh
python generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Wordpress/Scale/wordpress_scale.json

ls