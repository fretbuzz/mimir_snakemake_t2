#!/usr/bin/env bash

# note: this assumes that application is ALREADY deployed
# this just handles the stupid cycling that you have to do

while read instr; do

    docker service scale atsea_reverse_proxy=0 atsea_payment_gateway=0 atsea_database=0 atsea_appserver=0 atsea_visualizer=0;
    /bin/sleep 10;
    docker ps -q > containers.txt;

    while read p; do
        docker stop $p;
    done < containers.txt;

    /bin/sleep 10;

    docker service scale atsea_visualizer=1 atsea_reverse_proxy=5 atsea_payment_gateway=5 atsea_database=1 atsea_appserver=10;

    /bin/sleep 30;

    echo $instr;

    python run_experiment.py $instr;

    /bin/sleep 15;

done < meta_experiment_instructions.txt

