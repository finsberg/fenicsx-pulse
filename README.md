![_](docs/pulse-logo.png)

[![MIT](https://img.shields.io/github/license/finsberg/fenicsx-pulse)](https://github.com/finsberg/fenicsx-pulse/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/fenicsx-pulse.svg)](https://pypi.org/project/fenicsx_pulse/)
[![Test package](https://github.com/finsberg/fenicsx-pulse/actions/workflows/test_package_coverage.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/actions/workflows/test_package_coverage.yml)
[![Pre-commit](https://github.com/finsberg/fenicsx-pulse/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/actions/workflows/pre-commit.yml)
[![Deploy static content to Pages](https://github.com/finsberg/fenicsx-pulse/actions/workflows/build_docs.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/actions/workflows/build_docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Create and publish a Docker image](https://github.com/finsberg/fenicsx-pulse/actions/workflows/docker-image.yml/badge.svg)](https://github.com/finsberg/fenicsx-pulse/pkgs/container/fenicsx_pulse)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/fenicsx-pulse-coverage.json)](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/fenicsx-pulse-coverage.json)

# fenicsx-pulse

`fenicsx-pulse` is a cardiac mechanics solver based on FEniCSx. It is a successor of [`pulse`](https://github.com/finsberg/pulse) which is a cardiac mechanics solver based on FEniCS.

* Documentation: https://finsberg.github.io/fenicsx-pulse/
* Source code: https://github.com/finsberg/fenicsx-pulse

## Install

To install `fenicsx_pulse` you need to first [install FEniCSx](https://github.com/FEniCS/dolfinx#installation). Next you can install `fenicsx_pulse` via pip
```
python3 -m pip install fenicsx-pulse
```
We also provide a pre-built docker image with FEniCSx and `fenicsx_pulse` installed. You pull this image using the command
```
docker pull ghcr.io/finsberg/fenicsx-pulse:v0.2.0
```

## Getting started
Checkout out [the demos](https://finsberg.github.io/fenicsx-pulse/demo/unit_cube.html) in the documentation



## Contributing
See https://finsberg.github.io/fenicsx-pulse/CONTRIBUTING.html
