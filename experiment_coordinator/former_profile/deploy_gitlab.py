from experiment_coordinator.former_profile.kubernetes_setup_functions import *

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
	out = subprocess.check_output(["minikube", "addons", "enable", "ingress"])
	print out
	try:
		out = subprocess.check_output(["git", "clone", "https://gitlab.com/charts/gitlab.git"])
		print out
	except:
		print "gitlab directory already exists??"

	#out = subprocess.check_output(["cd", "./gitlab/examples"])
	#print out
	out = subprocess.check_output(["helm", "repo", "add", "gitlab", "https://charts.gitlab.io/"])
        print out
	out = subprocess.check_output(["helm", "repo", "update"])
        print out
	minikube_ip = subprocess.check_output(["minikube", "ip"])
	minikube_ip = minikube_ip.rstrip('\n')
	out = subprocess.check_output(["helm", "upgrade", "--install", "gitlabv2", "gitlab/gitlab","--timeout", "600", "--values", "./gitlab/examples/values-minikube.yaml","--set", "global.hosts.domain=example.com", "--set", "global.hosts.externalIP="+minikube_ip + ",global.hosts.domain="+minikube_ip+".nip.io"]) 
        print out
	out = subprocess.check_output(["kubectl", "get", "ingress"]) # the gitlab IP is the one you want to connect to
        print out

main()
