# replicating-simple-blackbox-attack

[![Coverage Status](https://coveralls.io/repos/github/EricKolibacz/replicating-simple-blackbox-attack/badge.svg?branch=main)](https://coveralls.io/github/EricKolibacz/replicating-simple-blackbox-attack?branch=main)
![Code Style Checks](https://github.com/EricKolibacz/replicating-simple-blackbox-attack/actions/workflows/code_style.yml/badge.svg)
![Python Tests](https://github.com/EricKolibacz/replicating-simple-blackbox-attack/actions/workflows/python_tests.yml/badge.svg)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repository replicates the code and results of the paper ["Simple Black-box Adversarial Attacks"](https://arxiv.org/pdf/1905.07121.pdf). The original code can be found [here](https://github.com/cg563/simple-blackbox-attack).


The idea is to not to blindly copy the official code of the authors, but to implement it independently. 

# Reference
* [The paper from Guo et al.](https://arxiv.org/pdf/1905.07121.pdf)
* Label index conversion from the official ImageNet to the indices which were used to pre-train the PyTorch ResNet50; by GitHub user raghakot [File](https://github.com/raghakot/keras-vis/tree/master/resources)/[Repo](https://github.com/raghakot/keras-vis)
