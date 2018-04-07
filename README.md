## Munnin
Munnin is an experimetnal apparatus designed to test the potential for anomaly-based data exfiltration detection in enterprise applications with a microservice architecture. It automates setting up the test microservice (currently weave's Sock Shop), generates background traffic, performs 'data exfiltration' (kinda), creates traffic matrices, and analyzes these traffic matrices to detect the aforementioned data exfiltration

## Motivation
The transition to Microsevice Architectures bring new challenges and opportunities to detecting data exfiltration. Current data exfiltration methods use keyword-based methods that don't work on encrypted traffic, necessitating the use of anomaly-based methods. Anomaly-based methods, however, don't work well in tradtional enterprise (e.g. 3-tier) application because of the lack of network-level visibility. The increased visibility (and granularity of that visibility) for microservice applications makes anomaly-based data exfiltration methods a possibility.

 ## Tech/framework used

<b>Built with</b>
Kubernetes orchestrator
Istio for metrics collection
Prometheus for metrics aggregation
Weave's Sock Shop as a sample application
Locust for background traffic simulation and data exfiltration simulation
Python's Pandas for Traffic Matrix Analysis

## Features
What makes your project stand out?

## Code Example
Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Installation
Provide step by step series of examples and explanations about how to get a development env running.

TODO: Update
Steps to use:
1. Install minikube/kubernetes
2. git clone weave's sock shop
3. go to the sock shop directory
4. clone the munnin directory
5. mkdir ./experimental_data/
6. python run_experiment.py
7. Hopefully some traffic matrixes will pop up

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests
Describe and show how to run the tests with code examples.

## How to use?
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.

## Contribute

Let people know how they can contribute into your project. A [contributing guideline](https://github.com/zulip/zulip-electron/blob/master/CONTRIBUTING.md) will be a big plus.

## Credits
Give proper credits. This could be a link to any repo which inspired you to build this project, any blogposts or links to people who contrbuted in this project. 

#### Anything else that seems useful

## License
A short snippet describing the license (MIT, Apache etc)

MIT © [Yourname]()
