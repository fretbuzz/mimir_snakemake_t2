
declare -i current_db_instances=24
declare -i goal_db_instances=50
declare -i current_wp_instances=1
declare -i goal_wp_instances=40

while [ $current_wp_instances -lt $goal_wp_instances ] 
do
current_wp_instances=$current_wp_instances+5
#current_db_instances=$current_db_instances+2
echo kubectl scale deploy wwwppp-wordpress "--replicas=$current_wp_instances"
kubectl scale deploy wwwppp-wordpress "--replicas=$current_wp_instances"
#kubectl scale statefulset my-release-pxc "--replicas=$current_db_instances"

sleep 240
done
