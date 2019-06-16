### SOCKOSHOP REPRODUCABILITY -- SCALE ###
## THIS SHOULD be all that I need to do to get the Sockshop reproducibility thing going ##

cd /mydata
export MINIKUBE_HOME=/mydata
sudo chown jsev /mydata
git clone https://github.com/fretbuzz/mimir_v2.git
cd ./mimir_v2/experiment_coordinator/
bash ../configs_to_reproduce_results/kubernetes_setup_script.sh

sudo python run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_exp.json ;\
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_100_mk2_exp.json ;\
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_120_exp.json;\
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_140_exp.json

sleep 60

bash ../configs_to_reproduce_results/cycle_minikube.sh

sleep 120

sudo python run_experiment.py --no_exfil --prepare_app --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_160_exp.json;\
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_40_exp.json ;\
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_60_exp.json ;\
sudo python run_experiment.py --no_exfil --config_file ../configs_to_reproduce_results/Data_Collection/Sockshop/Scale/sockshop_four_80_exp.json

sleep 60

cd ../analysis_pipeline/

python generate_paper_graphs.py --dont_update_config --config_json ../configs_to_reproduce_results/Data_Analysis/Sockshop/Scale/new_sockshop_scale.json

ls
####