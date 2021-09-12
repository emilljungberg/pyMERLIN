Motion Estimation and Retrospective correction Leveraging Interleaved Navigators - MERLIN
======================================================================================================

.. image:: https://img.shields.io/badge/License-MIT-green.svg
	:target: https://opensource.org/licenses/MIT


MERLIN is a method for Motion Estimation Retrospective correction Leveraging Interleaved Navigators. This repository contains python code (in ``pymerlin``) for performing motion correction on suitable MRI data. The example pipelines in ``pymerlin`` are specifically designed to be used with 3D radial, ZTE, acquisitions.


Installation
-----------------
Install the python tools for ``pymerlin`` with ``pip`` by cloning this repository and running the following command in the main folder::

	pip install -e .

Dependencies
-----------------
Besides the python dependencies listed in ``pymerlin/requirements.txt``, MERLIN has the following dependencies:

- `riesling <https://github.com/spinicist/riesling>`_ 3D non-cartesian reconstruction toolbox.
- `HD-BET <https://github.com/NeuroAI-HD/HD-BET>`_ for efficient brain extraction
- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_ for various processing

Usage
----------------
- Further details about the usage is found in the `documentation <https://pymerlin.readthedocs.io/en/latest/>`_.
- Examples and code to reproduce figures for paper submitted to MRM can be found in the `merlin_mrm <https://github.com/emilljungberg/merlin_mrm>`_ repository.
