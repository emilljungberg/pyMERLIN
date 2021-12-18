.. _Pipelines:

Pipelines
================

An example of how to use this command is provided in a repository for the MRM paper which can be found here:  `merlin_mrm <https://github.com/emilljungberg/merlin_mrm>`_

There are several steps involved in performing motion correction from the input k-space data to corrected image. ``pymerlin`` is designed to me modular and thus enabling the user to design a pipeline according to their needs. Below is an example of a pipeline for sliding window motion correction. I recommend using this as a start when you are designing your own motion correction pipeline.

An example of an end-to-end motion correction pipeline for MERLIN is provided in ``scripts/run_merlin_sw``. This is a bash script which takes a single ``.h5`` k-space file (in ``riesling`` format), a parameter file, and an output directory as input and runs through the necessary steps for motion correction.

.. code:: text

    usage: bash run_merlin_sw -i input_file -p par.txt -o moco_folder

    Required arguments
    -i      Input .h5 file       
    -o      Output folder
    -p      Paramter file

    Optional arguments
    -v      Verbose
    -h      Help 

There are a lot of settings needed for this script, and since you probably will find one set of settings that works for your data, the parameters are provided as a text file. You can create them using

.. code:: sh

    > pymerlin param [<your settings>] -o par.txt

This will give you all the available options for the motion correction. You can then run moco as

.. code:: sh

    > run_merlin_sw -i my_data.h5 -o moco_folder -p par.txt

The following steps are performed
    
    1. A directory for the motion correction is created where all the files will be saved.
    2. The k-space data is separated into interleaves using ``riesling split`` and saved in the ``interleaves`` folder. The number of interleaves is controlled by the number of spokes per interleave (option ``-n``) and the length of the sliding window (``--step``).
    3. Sensitivity maps are reconstructed using ``riesling sense`` from the low-res spokes.
    4. Navigators are reconstructed using conjugate gradient SENSE, using ``riesling cg``, and saved in ``navigators``. This loop is run in parallel, number of recon jobs per iteration can be controlled by ``--batchries``.
    5. A brain mask is automatically generated from the reference navigator using HD-BET and twice dilated to cover the head. Used for the registration process.
    6. Navigators are registered to the first navigator which is used as the reference. Registration is performed with ``pymerlin reg``, again in parallel. Results from the registration are saved as a pickle file in ``registrations``. 
    7. The registration objects are merged together to a single pickle ``all_reg_param.p`` using ``pymerlin merge``.
    8. Motion correction is applied to the input data using ``pymerlin moco`` and saved to ``<input_name>_moco.h5``.

The final step after this is to reconstruct the motion corrected image using your prefered method with ``riesling``. 
