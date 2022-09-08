## Description:

This installs the commonroad_rl repository inside a docker container.

## 1. instructions to build the image yourself
Make sure you have docker installed. Otherwise follow your platform-specific guide:
https://docs.docker.com/get-docker/

### 1.1 Download repositories and build docker container
Before you can start to build the docker, clone the commonroad-rl repository.
Start at the folder where you cloned the repo (like below) to include all files inside the docker build:

```bash
# clone the commonroad rl image
git clone https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl
# go into project root for installing the docker
cd commonroad_rl
sudo docker build --tag "commonroad_rl"  --file  ./commonroad_rl/install_docker/commonroad_rl.dockerfile .
```
The image built should now be completed.

### 1.2 Run the container
Run the following command to start the docker container:
```bash
sudo docker run -it -p 9000:8888 --name commonroad_rl_nb commonroad_rl 
```
confirm that the docker conatiner is running, i.e. `docker ps`. 
The port 9000 will expose the jupyter notebook.

### 1.3 Open Jupyter Notebook
You can now access the jupyter notebooks by opening `localhost:9000` in your browser.
If you have done the installation on a remote server /cluster, this might come in handy:

```
# map from remote port 9000 to port 8888 on local via ssh.
ssh -N -f -L localhost:9000:localhost:8888 youruser@yourserverip
```

