Polyase tutorial
================
Allelic gene and isoform expression analysis in potato
----------------------------------------------------------------
The jupyter notebook tutorial demonstrates how to use the **polyase** package to analyze allele-specific expression (ASE) in tetraploid potato using long-read RNA-seq data. The tutorial covers loading the data, calculating allelic ratios, filtering, statistical analysis, and visualization of ASE patterns.
The notebook is available `here <https://github.com/nadjano/polyase/blob/master/docs/source/potato_polyase.ipynb>`_. Files necessary for this tutorial are deposited on Zenodo `here <https://zenodo.org/records/17590760/files/polyase_tutorial_atlantic.zip?download=1&preview=1>`_.

Running the tutorial locally
-----------------------------
Create a conda environment that also includes ipykernel to run jupyter notebooks:

.. code-block:: bash

   conda create -n polyase python=3.12 ipykernel pip && conda activate polyase && pip install polyase && pip install pyranges

If you get an error regarding the installation of pyranges (a known issue on mac and windows),
then try to install it via conda:

.. code-block:: bash

   conda install -c bioconda pyranges

Running the tutorial within Docker
------------------------------------
A Docker image with PolyASE and JupyterLab is available on `Docker Hub <https://hub.docker.com/r/nadjano/polyase>`_.

1. Download the tutorial notebook and input data:

   - Notebook (download the raw file): https://github.com/NIB-SI/polyase/blob/master/docs/source/potato_polyase.ipynb
   - Input data: https://zenodo.org/records/17590760/files/polyase_tutorial_atlantic.zip?download=1

2. Unzip the input data and place the notebook and all data files into a local ``notebooks/`` folder.

3. Start the container, mounting your ``notebooks/`` folder:

   .. code-block:: bash

      docker run --rm -p 8888:8888 \
        -v $(pwd)/notebooks:/home/user/notebooks \
        nadjano/polyase:1.3.3

4. Open http://localhost:8888 in your browser, open ``potato_polyase.ipynb``, and update any file paths in the notebook to ``/home/user/notebooks/``.

.. raw:: html
   :file: _static/potato_polyase.html