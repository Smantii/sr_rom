# SR-ROM
This repository contains the code used to produce the results of the paper [Symbolic Regression of Data-Driven Reduced Order Model Closures for Under-Resolved, Convection-Dominated Flows
](https://arxiv.org/abs/2502.04703v1).

## Installation
The dependencies are collected in `environment.yaml` and can be installed, after cloning the repository, using [`mamba`]("https://github.com/mamba-org/mamba"):
```bash
$ mamba env create -f environment.yaml
```

Once the environment is installed and activated, install the library using

```bash
$ pip install .
```

## Usage
To reproduce the results, download the data available [here](https://drive.google.com/drive/folders/16rg1L3277Dp9IOpxF818vxinQVYy5bdn) and move them using

```bash
$ mv data/* src/sr_rom/data/
```


Then, change the results folder name in `compute_results.py` and run
```bash
$ python src/sr_rom/compute_results.py
```

## Citing
```
@article{manti2025symbolic,
  title={Symbolic Regression of Data-Driven Reduced Order Model Closures for Under-Resolved, Convection-Dominated Flows},
  author={Manti, Simone and Tsai, Ping-Hsuan and Lucantonio, Alessandro and Iliescu, Traian},
  journal={arXiv preprint arXiv:2502.04703},
  year={2025}
}
```
