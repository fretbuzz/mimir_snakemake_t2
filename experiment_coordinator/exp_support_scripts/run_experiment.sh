app_name=$1
cilium_p=$2
autoscale_p=$3
cpu_cutoff=$4

#minikube_ip=$2
#front_facing_port=$3
#config_file_name=$4
#exp_name=$5

rm -f /mydata/done_with_setup.txt

echo 'start run_experiment' >> /local/repository/run_experiment_note1.txt

bash ./kubernetes_setup.sh $cilium_p

echo 'start run_experiment n2' >> /local/repository/run_experiment_note2.txt

bash ./deploy_application.sh $app_name $autoscale_p $cpu_cutoff

echo 'start run_experiment n3' >> /local/repository/run_experiment_note3.txt

#python /mydata/mimir_v2/experiment_coordinator/run_experiment.py --exp_name $exp_name --config_file $config_file_name --prepare_app_p --port $front_facing_port -ip $minikube_ip --no_exfil

echo 'start run_experiment n4' >> /local/repository/run_experiment_note4.txt
echo 'done_with_that' >> /mydata/done_with_setup.txt
