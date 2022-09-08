#!/bin/bash

source ./setup_env.sh
nohup jupyter lab --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' >/dev/null 2>&1 &
