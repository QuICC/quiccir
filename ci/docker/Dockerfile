FROM docker.io/ubuntu:22.04

RUN apt-get -yqq update && apt-get -yqq upgrade \
 && apt-get -yqq install build-essential vim less \
 git cmake ninja-build python3-pip
#  \
#  && rm -rf /var/lib/apt/lists/*

RUN apt-get -yqq install lsb-release wget \
 software-properties-common gnupg

RUN wget https://apt.llvm.org/llvm.sh \
 && chmod +x llvm.sh \
 && ./llvm.sh 17

RUN apt-get -yqq install libmlir-17-dev mlir-17-tools

RUN pip3 install lit

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]
CMD ["/bin/bash"]
SHELL ["/bin/bash", "--rcfile", "/etc/profile", "-l", "-c"]