#!/usr/bin/env bash

exp_time_min=$1
hpa_file=$2

#hpa_file="${hpa_file//\" \"}"
#hpa_file="${hpa_file//\"\n"}"
echo ' ' > "$hpa_file"
for i in $(seq 1 "$((exp_time_min))");
do
    cur_hpa=$(kubectl get hpa --all-namespaces)
    #echo '"$cur_hpa" >> "$hpa_file"' # this is for testing purposes... should be reemoved eventually
    #echo "$cur_hpa"
    #echo "$hpa_file"
    echo "$cur_hpa" >> "$hpa_file"
    sleep 60
done
