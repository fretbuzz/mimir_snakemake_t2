from experiment_coordinator.former_profile.kubernetes_setup_functions import *

if __name__== "__main__":
  service_name = sys.argv[1]
  service_cluster_ip = get_svc_ip(service_name)
