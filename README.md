[![MIT](https://img.shields.io/github/license/finsberg/pulsex)](https://github.com/finsberg/pulsex/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/pulsex.svg)](https://pypi.org/project/pulsex/)
[![Test package](https://github.com/finsberg/pulsex/actions/workflows/test_package_coverage.yml/badge.svg)](https://github.com/finsberg/pulsex/actions/workflows/test_package_coverage.yml)
[![Pre-commit](https://github.com/finsberg/pulsex/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/pulsex/actions/workflows/pre-commit.yml)
[![Deploy static content to Pages](https://github.com/finsberg/pulsex/actions/workflows/build_docs.yml/badge.svg)](https://github.com/finsberg/pulsex/actions/workflows/build_docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Create and publish a Docker image](https://github.com/finsberg/pulsex/actions/workflows/docker-image.yml/badge.svg)](https://github.com/finsberg/pulsex/pkgs/container/pulsex)

# pulsex

`pulsex` is a cardiac mechanics solver based on FEniCSx. It is a successor of [`pulse`](https://github.com/finsberg/pulse) which is a cardiac mechanics solver based on FEniCS.

---

## Notice

**This repo is a complete rewrite of `pulse` to work with FEniCSx. The package is not yet ready for release.**

If you are using FEniCS please check out [`pulse`](https://github.com/finsberg/pulse) instead

---

* Documentation: https://finsberg.github.io/pulsex/
* Source code: https://github.com/finsberg/pulsex

## Install

To install `pulsex` you need to first [install FEniCSx](https://github.com/FEniCS/dolfinx#installation). Next you can install `pulsex` via pip
```
python3 -m pip install pulsex
```
We also provide a pre-built docker image with FEniCSx and `pulsex` installed. You pull this image using the command
```
docker pull ghcr.io/finsberg/pulsex:v0.1.1
```

## Simple Example

TBW


## Contributing

TBW
