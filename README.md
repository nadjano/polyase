<img style="width: 50%;" alt="polyASE_logo" src="https://github.com/user-attachments/assets/4d3dea85-c1a6-4707-a498-0e969a5e6c07" />

# PolyASE

A Python package for analyzing allele-specific expression in polyploid organisms.
This package is part of the [LongPolyASE](https://polyase.readthedocs.io/en/latest/index.html) framework for long-read RNA-seq allele-specific expression analysis in polyploid organisms.
Please see [syntelogfinder](https://github.com/NIB-SI/syntelogfinder) and [longrnaseq](https://github.com/NIB-SI/longrnaseq) for generating the input data for polyase.

<img style="width: 30%;" alt="polyase_analysis" src="https://github.com/user-attachments/assets/a334a3a9-ac6f-403e-8d3e-a2f4b4bbbb6d" />

## Installation

PolyASE is avialble on [PyPI](https://pypi.org/project/polyase/).
You can install polyase using pip:

<pre><code>pip install polyase</code></pre>

Pyranges must be installed manually for reading GTF files (most of the analysis can be run without pyranges).

To create a conda environemnt for running juypter notebook:

<pre><code>conda create -n polyase python=3.12 ipykernel pip && conda activate polyase && pip install polyase && pip install pyranges</code></pre>


## Docker

A Docker image with PolyASE and JupyterLab is available on Docker Hub:

```bash
docker run --rm -p 8888:8888 \
  -v $(pwd)/notebooks:/home/user/notebooks \
  nadjano/polyase:1.3.3
```

Then open http://localhost:8888 in your browser.

## Tutorial

A tutorial for analysis of allele-specific gene and isoform expression in tetraploid potato can be found [here](https://polyase.readthedocs.io/en/latest/tutorial_potato.html).
The input data for the tutorial can be found [here](https://zenodo.org/records/17590760/files/polyase_tutorial_atlantic.zip?download=1&preview=1).

### Running the tutorial within Docker

1. Download the tutorial notebook and input data:
   - Notebook (download the raw file): https://github.com/NIB-SI/polyase/blob/master/docs/source/potato_polyase.ipynb
   - Input data: https://zenodo.org/records/17590760/files/polyase_tutorial_atlantic.zip?download=1

2. Unzip the input data and place the notebook and all data files into a local `notebooks/` folder.

3. Start the container, mounting your `notebooks/` folder:

```bash
docker run --rm -p 8888:8888 \
  -v $(pwd)/notebooks:/home/user/notebooks \
  nadjano/polyase:1.3.3
```

4. Open http://localhost:8888 in your browser, open `potato_polyase.ipynb`, and update any file paths in the notebook to `/home/user/notebooks/`.

### Troubleshooting

if you get an error installing polyase with pip for some dependencies, e.g.:

- pyranges (a known issue on mac and windows).
- Failed to build llvmlit and numba

```
conda install -c conda-forge numba
conda install -c bioconda pyranges
```

## Citation
