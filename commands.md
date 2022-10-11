```bash
docker pull osrf/ros:melodic-desktop
docker pull nvidia/cuda:10.2-runtime-ubuntu18.04

docker run -it --network=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name=smmr osrf/ros:melodic-desktop bash

apt update ; apt upgrade -y ; apt install -y build-essential git vim curl wget python3-pip zsh; sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"; git clone https://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions

```
