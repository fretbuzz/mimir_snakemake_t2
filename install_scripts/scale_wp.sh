
declare -i current_db_instances=24
declare -i goal_db_instances=50
declare -i current_wp_instances=47
declare -i goal_wp_instnaces=150

while [ $current_db_instances -lt $goal_db_instances ] 
do
current_wp_instances=$current_wp_instances+7
current_db_instances=$current_db_instances+2
echo kubectl scale deploy wwwppp-wordpress "--replicas=$current_wp_instances"
kubectl scale deploy wwwppp-wordpress "--replicas=$current_wp_instances"
kubectl scale statefulset my-release-pxc "--replicas=$current_db_instances"

sleep 300
done
