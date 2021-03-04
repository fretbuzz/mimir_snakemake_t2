
currently this part works like this
# ssh onto node1
git clone https://github.com/fretbuzz/mimir_v2.git
bash ~/mimir_v2/configs_to_reproduce_results/full_cluster_scripts/first_part_of_setup.sh
# setup keys (see google doc)[https://docs.google.com/document/d/1iXs-iHZTkA5606_QG9VgkDqUHuLtMbNeusMeCDqNL7w/edit]
bash ~/mimir_v2/configs_to_reproduce_results/full_cluster_scripts/second_part_of_setup.sh
# do stuff now
bash ~/mimir_v2/configs_to_reproduce_results/full_cluster_scripts/third_part_of_setup.sh
# do stuff now
