FROM continuumio/anaconda3

RUN apt-get update && \
      apt-get -y -qq install sudo apt-utils

# copy 
COPY . /commonroad_rl

WORKDIR /commonroad_rl

RUN apt-get -y -qq install build-essential make cmake socat 

RUN conda env create -q -n cr37 -f environment.yml
RUN conda init bash

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cr37", "/bin/bash", "-c"]

# ---- START: just follows the readme.md at the root of commonroad-rl! --.
RUN pip install -q -r commonroad_rl/doc/requirements_doc.txt 
RUN conda install -q jupyter

RUN bash scripts/install.sh -e cr37

#run tests
RUN pytest commonroad_rl/tests --scope unit module -m "not slow"

# --- END: just follows the readme.md at the root of commonroad-rl! --.

ENTRYPOINT bash -c "\
conda activate cr37 &&\
cd /commonroad_rl/;\
socat TCP-LISTEN:8888,fork TCP:127.0.0.1:9000 &\
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 9000 "