from experiment_coordinator.kubernetes_setup_functions import *

def main():
	out = subprocess.check_output(["wget", "https://raw.githubusercontent.com/helm/helm/master/scripts/get"])
	print out
	out = subprocess.check_output(["chmod", "700", "get"])
	print out
	out = subprocess.check_output(["bash", "./get"])
	print out
	out = subprocess.check_output(["helm", "init"])
	print out
	time.sleep(5)
	wait_until_pods_done("kube-system") # need tiller pod deployed
	#out = subprocess.check_output(["/helm", "install", "--name", "wordpress", "stable/wordpress"])
	#print out
	try:
		out = subprocess.check_output(["helm", "install", "--name", "my-release", "--set", "mysqlRootPassword=secretpassword,mysqlUser=my-user,mysqlPassword=my-password,mysqlDatabase=my-database,replicas=15", "stable/percona-xtradb-cluster"])
		print out
	except:
		print "DB cluster must have already been initiated..."
	wait_until_pods_done("default") # wait until DB cluster is setup
	## TODO: setup wordpress servers then
	db_cluster_ip = get_svc_ip('my-release-pxc')
	print "db_cluster_ip", db_cluster_ip
	#helm install --name wwwppp --values /mydata/mimir/install_scripts/wordpress-values-production.yaml --set externalDatabase.host=10.103.42.190  stable/wordpress
	try:
		out = subprocess.check_output(["helm", "install", "--name", "wwwppp", "--values", "/mydata/mimir/install_scripts/wordpress-values-production.yaml", "--set", "externalDatabase.host=" + db_cluster_ip, "stable/wordpress"])
		print out
	except:
		print "wordpress deployment must already exist"

	num_wp_containers = 1
	goal_wp_containers = 1
	while num_wp_containers < goal_wp_containers:
		out = subprocess.check_output(["kubectl", "scale", "deploy", "wwwppp-wordpress", "--replicas=" + str(num_wp_containers)])
		num_wp_containers += 5
		wait_until_pods_done("default")
	

'''
out = subprocess.check_output(["helm", "install", "--name", "wordpress", "stable/wordpress"])
            print out
        wait_until_pods_done("default") # need new pods working before can start experiment
        if hpa:
            start_autoscalers(get_deployments('default'), '70')
        # helm install --name wordpress stable/wordpress
'''



main()
