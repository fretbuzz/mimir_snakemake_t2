## Munnin
Munnin is an experimetnal apparatus designed to test the potential for anomaly-based data exfiltration detection in enterprise applications with a microservice architecture. It automates setting up the test microservice (currently weave's Sock Shop), generates background traffic, performs 'data exfiltration' (kinda), creates traffic matrices, and analyzes these traffic matrices to detect the aforementioned data exfiltration

## Motivation
The transition to Microsevice Architectures bring new challenges and opportunities to detecting data exfiltration. Current data exfiltration methods use keyword-based methods that don't work on encrypted traffic, necessitating the use of anomaly-based methods. Anomaly-based methods, however, don't work well in tradtional enterprise (e.g. 3-tier) application because of the lack of network-level visibility. The increased visibility (and granularity of that visibility) for microservice applications makes anomaly-based data exfiltration methods a possibility.

 ## Tech/framework used

<b>Built with:</b>

* Kubernetes orchestrator

* Istio for metrics collection

* Prometheus for metrics aggregation

* Weave's Sock Shop as a sample application

* Locust for background traffic simulation and data exfiltration simulation

* Python's Pandas for Traffic Matrix Analysis

* Data visualization using matplotlib

## Features
It runs the experiment and helps you analyze the output! It's still pre-MVP stage, so there's not many other features!

## Code Example
Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Installation
WARNING: these may be old and aren't really tested, but I think they should work fine but no guarantees!

Steps to use:
1. Install minikube/kubernetes
2. git clone weave's sock shop
3. go to the sock shop directory
4. clone the munnin directory
5. mkdir ./experimental_data/
6. Make sure that you have python 2.7 with anaconda installed (b/c pandas)
7. python run_experiment.py
8. Some traffic matrixes will be stored in ./experimental_data in pickled form and some graphs should pop up, maybe some more stuff will happen when we get around to adding it :) 

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

For now, look at the top of the .py files for their API (they are far enough up that 'head' will print them)

## Tests
Describe and show how to run the tests with code examples.

For now, all the tests are in tests.py. Run with
python tests.py

Later on we might add some end-to-end tests that start a new instance of minikube and whatnot, but we'll be sure to put those in a new file. To be clear, the current tests rely on the information with ./tests and won't restart minikube / istio / sock shop.

## How to use?
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.

## Contribute

If you want to contribute, talk to fretbuzz. If you don't know who he/she is, then you should not have access to this repo. 

## Credits
Give proper credits. This could be a link to any repo which inspired you to build this project, any blogposts or links to people who contrbuted in this project. 

#### Anything else that seems useful

## License
At the moment, this code is under the Apache-2.0 License.

## Quick Design overview
run_experiment.py is essentially equivalent to a bash script that starts all the other stuff, but written in Python because we like python.

pull_from_prom.py runs during the experiment and pulls data from promtheus, parses them to create a traffic matrix, calculates the differential traffic matrix, and stores the result in a pickle file in ./experimental_data.

analyze_traffic_matrixes.py unpickles those traffic matrixes and computes useful statistics. It includes functionality to walk through the matrix timestamp-by-timestamp, in order to simulate recieving those values live.

parameters.py contains most of the useful experimental parameters. The eventual goal here is to only need to modify this file in order to run a bunch of different experiments.
