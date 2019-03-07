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
  kubectl create -f /local/repository/sock-shop-ns.yaml -f /local/repository/modified_complete-demo.yaml
  bash /local/repository/scale_sockshop.sh
elif [ "$1" = "drupal" ]; then
  echo "see, it was drupal" >> /local/repository/deploy_test.txt
fi
