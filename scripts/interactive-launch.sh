#!/bin/bash
BASEDIR=${BASEDIR:-$(dirname $(cd $(dirname ${BASH_SOURCE[0]}) && pwd))}
set -a  # mark all variables below as exported (environment) variables

# Indentify this script as source of job configuration
K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}

K8S_NUM_GPU=1  # max of 2 (contact ETS to raise limit)
K8S_NUM_CPU=8  # max of 8 ("")
K8S_GB_MEM=64 # max of 64 ("")

# Controls whether an interactive Bash shell is started
SPAWN_INTERACTIVE_SHELL=YES

# Sets up proxy URL for Jupyter notebook inside 
PROXY_ENABLED=NO

#K8S_ENTRYPOINT="/bin/sleep"
#K8S_ENTRYPOINT_ARGS_EXPANDED='"8640000"'
K8S_DOCKER_IMAGE="ghcr.io/ucsd-ets/scipy-ml-notebook:2025.1-stable" #"ucsdets/donkeycar-notebook:2021.4.3-kernel-dev"
#K8S_DOCKER_IMAGE="ghcr.io/ucsd-ets/donkeycar-notebook:fa22"

if [ $(basename "$0") == "launch-148-gpu.sh" ]; then
	K8S_NUM_GPU=1  # max of 2 (contact ETS to raise limit)
fi
#exec ${BASEDIR}/bin/launch.sh "$@"
#-J -P "@"
exec launch.sh "$@" -i $K8S_DOCKER_IMAGE -c $K8S_NUM_CPU -m $K8S_GB_MEM -g $K8S_NUM_GPU
