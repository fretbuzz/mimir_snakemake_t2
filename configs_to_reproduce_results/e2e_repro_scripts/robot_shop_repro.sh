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

  # then do robot-shop-specific-dependencies
  . ../configs_to_reproduce_results/setup_robot_shop.sh

  # TODO: run experiments here...

fi


cd ../analysis_pipeline/

. ../configs_to_reproduce_results/install_mimir_depend_scripts.sh

mkdir multilooper_outs

# TODO: re-enable when ready
#sudo python -u generate_paper_graphs.py --config_json ../configs_to_reproduce_results/Data_Analysis/Wordpress/Scale/wordpress_scale.json | tee ./multilooper_outs/wordpress_scale.log

ls