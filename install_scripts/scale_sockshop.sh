
declare -i current_carts_instances=1
declare -i goal_carts_instnaces=10

kubectl scale deploy catalogue orders payment queue-master shipping user front-end --replicas=10 --namespace="sock-shop"
while [ $current_carts_instances -lt $goal_carts_instances ] 
do
current_carts_instances=$current_carts_instances+3
kubectl scale deploy carts "--replicas=$current_carts_instances" --namespace="sock-shop"

sleep 360
done
