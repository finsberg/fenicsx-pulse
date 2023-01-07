# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

ARG REPO_BRANCH="main"
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# Install pip and git with apt
RUN apt-get update && \
    apt-get install -y python3-pip git

# We upgrade pip and setuptools
RUN python3 -m pip install pip setuptools --upgrade


# We set the working directory to install docker dependencies
WORKDIR /tmp/

# Install the Python-module
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/finsberg/fenicsx-pulse
RUN python3 -m pip install  --no-cache-dir ./fenicsx-pulse[all]


# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
