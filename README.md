# corrgi

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/corrgi?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/corrgi/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks-mask-incubator/corrgi/smoke-test.yml)](https://github.com/lincc-frameworks-mask-incubator/corrgi/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/lincc-frameworks-mask-incubator/corrgi/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks-mask-incubator/corrgi)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/lincc-frameworks-mask-incubator/corrgi/asv-main.yml?label=benchmarks)](https://lincc-frameworks-mask-incubator.github.io/corrgi/)

This project provides glue code between [LSDB](https://github.com/astronomy-commons/lsdb) and [gundam](https://github.com/lincc-frameworks-mask-incubator/gundam) to perform fast calculation of 2-point correlation functions.

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
```