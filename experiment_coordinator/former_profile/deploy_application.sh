echo 'this is a test!'
echo $1
autoscale_p=$2
cpu_cutoff=$3

echo "see, it kinda worked" >> /local/repository/deploy_test_prior.txt
echo "$1" >> /local/repository/deploy_test_prior_val.txt
echo "$0" >> /local/repository/deploy_test_prior_val_two.txt
echo "$2" >> /local/repository/deploy_test_prior_val_three.txt

if [ "$1" = "wordpress" ]; then
  echo "it was testtest"
  echo "see, it was wordpress" >> /local/repository/deploy_test.txt
  # deploy_wordpress.py
  bash /mydata/mimir_snakemake_t2/experiment_coordinator/install_scripts/install_selenium_dependencies.sh
  if [ -z "autoscale_p" ]
  then
      python /mydata/mimir_snakemake_t2/experiment_coordinator/former_profile/deploy_wordpress.py
  else
      python /mydata/mimir_snakemake_t2/experiment_coordinator/former_profile/deploy_wordpress.py --autoscale_p --cpu_cutoff=$cpu_cutoff
  fi
elif [ "$1" = "eShop" ]; then
  echo "see, it was eShop" >> /local/repository/deploy_test.txt
 # deploy_eshop.sh
elif [ "$1" = "gitlab" ]; then
  echo "see, it was gitlab" >> /local/repository/deploy_test.txt
 # deploy_gitlab.py
elif [ "$1" = "sockshop" ]; then
  echo "see, it was sockshop" >> /local/repository/deploy_test.txt
  kubectl create -f ./sock-shop-ns.yaml -f ./sockshop_modified.yaml
  bash ./scale_sockshop.sh

  if [ -n "$autoscale_p" ]; then
    # activate autoscaling
    minikube addons enable heapster
    minikube addons enable metrics-server
    git clone https://github.com/microservices-demo/microservices-demo.git
    kubectl apply -f ./microservices-demo/deploy/kubernetes/autoscaling/
  fi
elif [ "$1" = "drupal" ]; then
  echo "see, it was drupal" >> /local/repository/deploy_test.txt
elif [ "$1" = "hipster" ]; then
  echo "see, it was hipsterStore" >> /local/repository/deploy_test.txt
  bash /mydata/mimir_snakemake_t2/experiment_coordinator/former_profile/deploy_hipsterStore.sh
  if [ -z "autoscale_p" ]
  then
      echo "WARNING: NEED TO DO MORE WORK ON NON-AUTOSCALING HIPSTER"
  else
      bash /mydata/mimir_snakemake_t2/experiment_coordinator/former_profile/autoscale_hipsterStore.sh
  fi
fi
