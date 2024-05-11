#!/usr/bin/env bash

set -x

if [[ $# -lt 2 ]] ; then
    echo 'too few arguments supplied'
    exit 1
fi

NAME=$1
OPTIONS=$2

srun \
    --job-name=Tp2m \
    --gres=gpu:8 \
    --ntasks=1 \
    --kill-on-bad-exit=1 \
    python entrypoint_train.py --name ${NAME} --options ${OPTIONS} &
