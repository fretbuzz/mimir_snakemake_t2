## Mimir
Mimir is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants.


## Running Analysis Pipeline
Currently, it only works off-line (with network pcap files). Add the desired configuration to analysis_pipeline/pipeline_recipes.py and that'll run it (see current contents of file for usage example). Note: If your thinking of running this yourself, you should probably talk to fretbuzz first.

Detailed setup/running instructions are being worked on at the moment and should be available soon. If you need to run it before then, send fretbuzz a message.

## Running Experimental Coordinator

The experimental coordinator handles simulating traffic/exfiltration on a microservic deployment. There are several support scripts to setup the applications and a coordinator that handles simulating user traffic, simulates exfiltration, and collects all the relevant data, including a pcap of all network activity on the cluster.

NOTE: at the moment, simulating exfiltration is NOT supported (support will be added within the next week)
NOTE: tested on Ubuntu 16.04.1 LTS

# Step 1: Install Minikube
Minikube is a local kubernetes cluster. The microservice applications will be deployed onto the Minikube cluster. The official installation instructions can be found here: https://kubernetes.io/docs/tasks/tools/install-minikube/

# Step 2: Start Minikube
I recommend starting minikube with at least 2 cpus (ideally 4), 8 gigabytes of memory, and 25 gigabytes of disk space
	e.g., minikube start --memory 8192 --cpus=4 --disk-size 25g
 Then need to enable necessary addons:
    minikube addons enable heapster
    minikube addons enable metrics-server
 
# Step 3: Deploy Relevant Microservice Application
 The currently supported applications are Sockshop, a Wordpress deployment, and HipsterStore (within the next weeek). Deployment instructions vary per application.

Sockshop: (a) deploy delpoyments and services: kubectl apply -f ./experimental_coordiantor/sockshop_setup/sock-shop-ns.yaml -f ./experimental_coordiantor/sockshop_setup/sockshop_modified.yaml
          (b) enable autoscaling: git clone https://github.com/microservices-demo/microservices-demo.git
                                  kubectl apply -f ./microservices-demo/deploy/kubernetes/autoscaling/
                                  
Wordpress: Two options to deploy: Option 1: Deploy manually
           (a) Install helm via instructions here: https://helm.sh/docs/using_helm/#installing-helm
			        (b) Start helm: helm init
			        (c) Install db cluster: helm install --name my-release --set mysqlRootPassword=secretpassword,mysqlUser=my-user,mysqlPassword=my-password,mysqlDatabase=my-database,replicas=3 table/percona-xtradb-cluster
			   		            Note: can modify the number of replicas as desired
			   	    (d) Wait until all the pods of the db cluster are installed
           (e) Start wordpress servers: helm install --name wwwppp --values /mydata/mimir_v2/experiment_coordinator/wordpress_setup/wordpress-values-production.yaml --set externalDatabase.host=DB_CLUSTER_IP stable/wordpress
      NOTE: DB_CLUSTER_IP needs to be replaced by the ip of the db cluster. This can by found using the command 'kubectl get svc' and looking at the IP of the 'my-release-pxc' service.
           (f) Enable autoscaling of wordpress servce: kubectl autoscale deployment wwwppp-wordpress --min=1 --max=10--cpu-percent=80
           NOTE: might want to modify min/max pods amounts depending on system capabilities
           (g) wait until all the pods are finished deploying
          Option 2: deploy using convenience script: 
              python ./experimental_coordiantor/wordpress_setup/ --autoscale_p
           Note: this might NOT work, in which case you'd have to default to the previous list of commands

HipsterStore: (a) install skaffold: https://skaffold.dev/docs/getting-started/#installing-skaffold
              (b) clone repo: git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
                              cd ./microservices-demo
              (c) deploy using skaffold: skaffold run 
                     NOTE: this'll take a while to run the first timee (~ 20 min)


# Step 4: Install Experimental Coordinator Dependencies

[TODO]

# Step 5: Configure Experimental Parameters

[TODO]

# Step 6: Start Experiment

[TODO]
 
