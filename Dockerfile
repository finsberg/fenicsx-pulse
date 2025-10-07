# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/lab:stable

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install -r https://raw.githubusercontent.com/scientificcomputing/scifem/refs/heads/main/build-requirements.txt
RUN python3 -m pip install scifem --no-build-isolation
RUN python3 -m pip install ".[docs]"
