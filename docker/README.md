### Docker installation

## Build image
To build the docker image on your own machine, which may take 15-30 mins:
```
docker build -t stable-dreamfusion:latest .
```

If you have the error **No CUDA runtime is found** when building the wheels for tiny-cuda-nn you need to setup the nvidia-runtime for docker.
```
sudo apt-get install nvidia-container-runtime
```
Then edit `/etc/docker/daemon.json` and add the default-runtime:
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
And restart docker:
```
sudo systemctl restart docker
```
Now you can build tiny-cuda-nn inside docker.

## Download image
To download the image (~6GB) instead:
```
docker pull supercabb/stable-dreamfusion:3080_0.0.1
docker tag supercabb/stable-dreamfusion:3080_0.0.1 stable-dreamfusion
```

## Use image

You can launch an interactive shell inside the container:

```
docker run --gpus all -it --rm -v $(cd ~ && pwd):/mnt stable-dreamfusion /bin/bash
```
From this shell, all the code in the repo should work.

To run any single command `<command...>` inside the docker container:
```
docker run --gpus all -it --rm -v $(cd ~ && pwd):/mnt stable-dreamfusion /bin/bash -c "<command...>"
```
To train:
```
export TOKEN="#HUGGING FACE ACCESS TOKEN#"
docker run --gpus all -it --rm -v $(cd ~ && pwd):/mnt stable-dreamfusion /bin/bash -c "echo ${TOKEN} > TOKEN && python3 main.py --text \"a hamburger\" --workspace trial -O"

```
Run test without gui:
```
export PATH_TO_WORKSPACE="#PATH_TO_WORKSPACE#"
docker run --gpus all -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v $(cd ~ && pwd):/mnt -v $(cd ${PATH_TO_WORKSPACE} && pwd):/app/stable-dreamfusion/trial stable-dreamfusion /bin/bash -c "python3 main.py --workspace trial -O --test"
```
Run test with gui:
```
export PATH_TO_WORKSPACE="#PATH_TO_WORKSPACE#"
xhost +
docker run --gpus all -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v $(cd ~ && pwd):/mnt -v $(cd ${PATH_TO_WORKSPACE} && pwd):/app/stable-dreamfusion/trial stable-dreamfusion /bin/bash -c "python3 main.py --workspace trial -O --test --gui"
xhost -
```







