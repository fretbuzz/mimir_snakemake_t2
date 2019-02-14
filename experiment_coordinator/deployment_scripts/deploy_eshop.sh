echo "first some logistics..."
git clone https://github.com/dotnet-architecture/eShopOnContainers.git
cd ./eShopOnContainers/k8s/helm

echo "installing eShopOnContainers..."

appName="eshop"
imageTag="latest"
dns="192.168.39.56" # todo: get programmatically
#dns="eshop"
#dns=""

declare -a infras=("sql-data" "nosql-data" "rabbitmq" "keystore-data" "basket-data")
declare -a charts=("apigwmm" "apigwms" "apigwwm" "apigwws" "basket-api" "catalog-api" "identity-api" "locations-api" "marketing-api" "mobileshoppingagg" "ordering-api" "ordering-backgroundtasks" "ordering-signalrhub" "payment-api" "webmvc" "webshoppingagg" "webspa" "webstatus")

helm repo update

for infra in ${infras[@]}; do
	echo "Installing $infra"
	echo "helm install --values app.yaml --values inf.yaml --values ingress_values.yaml --set app.name=$appName --set inf.k8s.dns=$dns --name="$appName-$infra" $infra"
	helm install --values app.yaml --values inf.yaml --values ingress_values.yaml --set app.name=$appName --set inf.k8s.dns=$dns --name="$appName-$infra" $infra     
done

for chart in ${charts[@]}; do
	echo "Installing $chart"
	echo helm install --values app.yaml --values inf.yaml --values ingress_values.yaml --set app.name=$appName --set inf.k8s.dns=$dns --set image.tag=$imageTag --set image.pullPolicy=Always --name="$appName-$chart" $chart
	helm install --values app.yaml --values inf.yaml --values ingress_values.yaml --set app.name=$appName --set inf.k8s.dns=$dns --set image.tag=$imageTag --set image.pullPolicy=Always --name="$appName-$chart" $chart 
done

echo "eShop installed"
