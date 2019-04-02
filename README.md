## Mimir
Mimir is an experimental apparatus designed to test the potential for anomaly-based data exfiltration detection in microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants.


## Running Analysis Pipeline
Currently, it only works off-line (with network pcap files). Add the desired configuration to analysis_pipeline/pipeline_recipes.py and that'll run it (see current contents of file for usage example). Note: If your thinking of running this yourself, you should probably talk to fretbuzz first.

Detailed setup/running instructions are being worked on at the moment and should be available soon. If you need to run it before then, send fretbuzz a message.

## Running Experimental Coordinator

The experimental coordinator handles simulating traffic/exfiltration on a microservic deployment. There are several support scripts to setup the applications and a coordinator that handles simulating user traffic, simulates exfiltration, and collects all the relevant data, including a pcap of all network activity on the cluster.

NOTE: at the moment, simulating exfiltration is NOT supported (support will be added within the next week)

NOTE: tested on Ubuntu 16.04.1 LTS

#### Step 1: Install Minikube
Minikube is a local kubernetes cluster. The microservice applications will be deployed onto the Minikube cluster. The official installation instructions can be found here: https://kubernetes.io/docs/tasks/tools/install-minikube/

#### Step 2: Start Minikube
I recommend starting minikube with at least 2 cpus (ideally 4), 8 gigabytes of memory, and 25 gigabytes of disk space
	e.g., <pre><code> minikube start --memory 8192 --cpus=4 --disk-size 25g </code></pre>
	
 Then need to enable necessary addons:
    <pre><code>  minikube addons enable heapster </code></pre>
    <pre><code>  minikube addons enable metrics-server </code></pre>
 
#### Step 3: Deploy Relevant Microservice Application
 The currently supported applications are Sockshop, a Wordpress deployment, and HipsterStore (within the next weeek). Deployment instructions vary per application. Only one of these should be setup at a given time.

Sockshop: 

<pre><code>
(a) deploy delpoyments and services: kubectl apply -f ./experimental_coordiantor/sockshop_setup/sock-shop-ns.yaml -f ./experimental_coordiantor/sockshop_setup/sockshop_modified.yaml

(b) enable autoscaling: git clone https://github.com/microservices-demo/microservices-demo.git
                        kubectl apply -f ./microservices-demo/deploy/kubernetes/autoscaling/
</code></pre>

Wordpress: Two options to deploy: 
Option 1: deploy using convenience script (you should probably read it before using though): 
<pre><code>	  
python ./experimental_coordiantor/wordpress_setup/ --autoscale_p
</code></pre>
	      
Note: this might NOT work, in which case you'd have to default to the following manual deployment

Option 2: Deploy manually:
<pre><code>
(a) Install helm via instructions here: https://helm.sh/docs/using_helm/#installing-helm
	   
(b) Start helm: helm init
				
(c) Install db cluster: helm install --name my-release --set mysqlRootPassword=secretpassword,mysqlUser=my-user,mysqlPassword=my-password,mysqlDatabase=my-database,replicas=3 table/percona-xtradb-cluster
				
Note: can modify the number of replicas as desired
						    
(d) Wait until all the pods of the db cluster are installed
				    
(e) Start wordpress servers: helm install --name wwwppp --values experiment_coordinator/wordpress_setup/wordpress-values-production.yaml --set externalDatabase.host=DB_CLUSTER_IP stable/wordpress
	   
NOTE: DB_CLUSTER_IP needs to be replaced by the ip of the db cluster. This can by found using the command 'kubectl get svc' and looking at the IP of the 'my-release-pxc' service.
      
(f) Enable autoscaling of wordpress servce: kubectl autoscale deployment wwwppp-wordpress --min=1 --max=10--cpu-percent=80
	   
NOTE: might want to modify min/max pods amounts depending on system capabilities
	   
(g) wait until all the pods are finished deploying
</code></pre>

HipsterStore: 
<pre><code>
(a) install skaffold: https://skaffold.dev/docs/getting-started/#installing-skaffold

(b) clone repo: git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
	      
cd ./microservices-demo
			      
(c) deploy using skaffold: skaffold run 
</code></pre>
NOTE: this'll take a while to run the first time (~ 20 min)


#### Step 4: Install Experimental Coordinator Dependencies

Install Python Dependencies: pip install -r experiment_coordinator/requirements.txt

Wordpress Only: Setup of the application is handled using a selenium script. Therefore selenium must be installed. Specifically, the Firefox WebDriver MUST be used. Here's a good set of instructions: 

https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Cross_browser_testing/Your_own_automation_environment

Follow the "Setting up Selenium in Node" section, but skip everything to do with javascript/node. (or can google to find alternative setup instructions). 

#### Step 5: Configure Experimental Parameters

Some experimental parameters need to be configured before starting the experiment. See the example in experiment_coordinator/experimental_configs/sockshop_example.json

Note: There's an analogous example for wordpress at experiment_coordinator/experimental_configs/wordpress_example.json

The various fields need to be filled out appropriately. I'll now go through what all the fields mean.

"application_name": name of the application that was previously setup (either sockshop or wordpress)

"experiment_name": used for saving the results

"path_to_docker_machine_tls_certs": location of minikube TLS certs (needed to communicate with VM); should be located at the location of the minikube installation /.certs

"experiment_length_sec": how long the experiment should last (in seconds)

These values are related to preparing the application before simulating user traffic. Typically this is used to pre-load the DB with data.

"setup": "number_customer_records": number of customer records to create

"setup": "number_background_locusts": number of background locusts to generate the customer records

"setup": "background_locust_spawn_rate": spawn rate of the background locusts (per second)

These values are related to simualting user traffic during the experiment.

"experiment: "number_background_locusts": number of background locusts to generate user traffic (each locust is roughly one customer)

"experiment: "background_locust_spawn_rate": spawn raet of the background locusts (per second)

#### Step 6: Start Experiment

We'll now start the experimental coordinator. It handles collecting the necessary data, simulating background traffic, and simulating exfiltration (though simulating exfiltration is currently not supported).

Move to the experimental apparatus directory. 

python run_experiment.py --exp_name [name of experiment] --config_file [path to config file prepared in previous step] --prepare_app_p --port [port of exposed service] --ip [ip of minikube] --no_exfil

Note: the port of the exposed service can be found by: <code><pre> kubectl get svc --all-namespaces </code></pre> and then looking for front-end for sockshop or wwpp-wordpress for wordpress

Note: the ip of minikube can be found with: <code><pre> minikube ip </code></pre>

Note: name of experiment does not have to agree with what's in the config file.
 
