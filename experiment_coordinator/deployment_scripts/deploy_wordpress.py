from experiment_coordinator.former_profile.kubernetes_setup_functions import *
import argparse

def main(autoscale_p=False, cpu_percent_cuttoff=80):
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
        out = subprocess.check_output(["helm", "install", "--name", "my-release", "--set", "mysqlRootPassword=secretpassword,mysqlUser=my-user,mysqlPassword=my-password,mysqlDatabase=my-database,replicas=7", "stable/percona-xtradb-cluster"])
        print out
    except:
        print "DB cluster must have already been initiated..."
    wait_until_pods_done("default") # wait until DB cluster is setup
    ## TODO: setup wordpress servers then
    db_cluster_ip = get_svc_ip('my-release-pxc')
    print "db_cluster_ip", db_cluster_ip
    #helm install --name wwwppp --values /mydata/mimir/install_scripts/wordpress-values-production.yaml --set externalDatabase.host=10.103.42.190  stable/wordpress
    try:
        out = subprocess.check_output(["helm", "install", "--name", "wwwppp", "--values", "/mydata/mimir_snakemake_t2/experiment_coordinator/install_scripts/wordpress-values-production.yaml", "--set", "externalDatabase.host=" + db_cluster_ip, "stable/wordpress"])
        print out
    except:
        print "wordpress deployment must already exist"

    num_wp_containers = 1
    goal_wp_containers = 23
    while num_wp_containers < goal_wp_containers:
        out = subprocess.check_output(
            ["kubectl", "scale", "deploy", "wwwppp-wordpress", "--replicas=" + str(num_wp_containers)])
        num_wp_containers += 5
        wait_until_pods_done("default")

    if autoscale_p:
        heapstr_str = "minikube addons enable heapster"
        out = subprocess.check_output(heapstr_str)
        print "heapstr_str_response ", out
        metrics_server_str= "minikube addons enable metrics-server"
        out = subprocess.check_output(metrics_server_str)
        print "metrics_server_str_response ", out
        wait_until_pods_done("kube-system")

        autoscale_cmd_str = "kubectl autoscale deployment wwwppp-wordpreses --min=" + str(15) + " --max= " + str(
            goal_wp_containers) + " --cpu-percent=" + str(cpu_percent_cuttoff)
        out = subprocess.check_output(autoscale_cmd_str)
        print "autoscale_out: ", out

'''
out = subprocess.check_output(["helm", "install", "--name", "wordpress", "stable/wordpress"])
            print out
        wait_until_pods_done("default") # need new pods working before can start experiment
        if hpa:
            start_autoscalers(get_deployments('default'), '70')
        # helm install --name wordpress stable/wordpress
'''


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='deploys the wordpress application')

    parser.add_argument('--cpu_cutoff',dest="cpu_cutoff", default='80')
    parser.add_argument('--autoscale_p', dest='prepare_app_p', action='store_true',
                        default=False,
                        help='should we autoscale the wordpress service')
    args = parser.parse_args()

    main(autoscale_p=args.autoscale_p, cpu_percent_cuttoff=args.cpu_cutoff)
#--autoscale_p=$autoscale_p --cpu_cutoff=$cpu_cutoff