[![MIT](https://img.shields.io/github/license/finsberg/pulsex)](https://github.com/finsberg/pulsex/blob/main/LICENSE)
# pulsex

`pulsex` is a cardiac mechanics solver based on FEniCSx. It is a successor of [`pulse`](https://github.com/finsberg/pulse) which is a cardiac mechanics solver based on FEniCS.

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
