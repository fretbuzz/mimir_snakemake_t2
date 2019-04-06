import subprocess
from kubernetes_setup_functions import *

def scale_wordpress(autoscale_p, cpu_percent_cuttoff, deployment_scaling):
    goal_wp_containers = deployment_scaling['wordpress']['max']
    wp_min = deployment_scaling['wordpress']['min']
    wp_cpu_percent_cutoff = cpu_percent_cuttoff

    if autoscale_p and cpu_percent_cuttoff != '':
        heapstr_str = ["minikube", "addons", "enable", "heapster"]
        out = subprocess.check_output(heapstr_str)
        print "heapstr_str_response ", out
        metrics_server_str= ["minikube", "addons", "enable", "metrics-server"]
        out = subprocess.check_output(metrics_server_str)
        print "metrics_server_str_response ", out
        wait_until_pods_done("kube-system")

        autoscale_cmd_str = ["kubectl", "autoscale", "deployment", "wwwppp-wordpress", "--min=" + str(wp_min), "--max=" + str(
            goal_wp_containers),"--cpu-percent=" + str(wp_cpu_percent_cutoff)]
        out = subprocess.check_output(autoscale_cmd_str)
        print "autoscale_out: ", out
        wait_until_pods_done("default")
    else:
        num_wp_containers = 1
        while num_wp_containers < goal_wp_containers:
            out = subprocess.check_output(
                ["kubectl", "scale", "deploy", "wwwppp-wordpress", "--replicas=" + str(num_wp_containers)])
            num_wp_containers += 5
            wait_until_pods_done("default")

def deploy_wp(deployment_scaling):
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

    try:
        out = subprocess.check_output(["helm", "install", "--name", "my-release", "--set", "mysqlRootPassword=secretpassword,mysqlUser=my-user,mysqlPassword=my-password,mysqlDatabase=my-database,replicas=" + str(deployment_scaling['pxc']["max"]), "stable/percona-xtradb-cluster"])
        print out
    except:
        print "DB cluster must have already been initiated..."

    wait_until_pods_done("default") # wait until DB cluster is setup
    db_cluster_ip = get_svc_ip('my-release-pxc')

    print "db_cluster_ip", db_cluster_ip

    try:
        out = subprocess.check_output(["helm", "install", "--name", "wwwppp", "--values", "./wordpress_setup/wordpress-values-production.yaml", "--set", "externalDatabase.host=" + db_cluster_ip, "stable/wordpress"])
        print out
    except:
        print "wordpress deployment must already exist"