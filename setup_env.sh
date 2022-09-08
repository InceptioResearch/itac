#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export ITAC_ROOT=$(pwd)
export PYTHONPATH=${ITAC_ROOT}:${ITAC_ROOT}/itac

echo '[ITAC_ROOT] : ' ${ITAC_ROOT}
echo '[PYTHONPATH] : ' ${PYTHONPATH}
