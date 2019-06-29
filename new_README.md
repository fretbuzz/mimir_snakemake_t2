# MIMIR: Graph-based Data Exfiltration Detection for Microservices

## Table of Contents
+ [About](#about)
+ [Getting Started](#getting_started)
+ [Usage](#usage)
+ [Contributing](../CONTRIBUTING.md)
+ [Reproducing Graphs from Paper](#repro)
+ [FAQ](#FAQ)

## About <a name = "about"></a>
This repository contains a prototype implementation of a graphical method for detecting data exfiltration in 
microservice-architecture applications. It creates a graphical representation of network communication and flags deviations from structural invariants. 
The goal is to detect data exfiltration, but it should also be effective at detecting other types of anomalous traffic, such as port scans or lateral movements by attackers.

This analysis pipeline been tested on Ubuntu 16.04. 

## Getting Started <a name = "getting_started"></a>
These instructions will walk you through setting up the analysis pipeline and running the demo. The demo consists of a
pre-trained model and some provided network activity data. 

See [Reproducing Graphs from Paper](#repro) for how to reproduce the 
graphs from the paper draft. See [Collecting New Data](#collecting data) for how to collect additional
network activity data and [Training New Model](#train_new_model) for how to train a new model.

### Prerequisites



What things you need to install the software and how to install them...

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## Usage <a name = "usage"></a>

Add notes about how to use the system.

### Training New Model <a name = "train_new_model"></a>

### Run Model on New Data <a name = "run_on_new_data"></a>

## Reproducing Graphs from Paper <a name = "repro"></a>
\[TODO\]

## FAQ <a name = "FAQ"></a>
\[TODO\]

### Collecting New Data <a name = "collecting data"></a>