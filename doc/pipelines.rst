.. _Pipelines:

Pipelines
================
There are several steps involved in performing motion correction from the input k-space data to corrected image. ``pymerlin`` is designed to me modular and thus enabling the user to design a pipeline according to their needs. Below is an example of a pipeline for sliding window motion correction. I recommend using this as a start when you are designing your own motion correction pipeline.

An example of an end-to-end motion correction pipeline for MERLIN is provided in ``scripts/run_merlin_sw``. This is a bash script which takes a single ``.h5`` k-space file (in ``riesling`` format) as input and runs through the necessary steps for motion correction. It has several options which are explained by the help

.. code:: text

    usage: bash run_merlin -i input_file -n spokes_per_int

    Required arguments
    -i      | --input       
    -n      | --nspokes

    Optional arguments
    --out           | Output folder (./<input_mocodir>)
    --thr           | Navigator image threshold (300)
    --its           | CG SENSE iterations (8)
    --ds            | Navigator downsampling (3)
    --fov           | MOCO field of view (240)
    --ow            | Overwrite files without asking
    --gap           | Set gap for Zinfandel
    --step          | Step size for sliding window
    --ref           | Reference navigator
    --batchitk      | Batch size for parallel ITK
    --batchries     | Batch size for parallel riesling
    --threaditk     | Number of threads for ITK
    --threadries    | Number of threads for riesling
    -v              | --verbose (0)
    -h              | --help 

The following steps are performed
    
    1. A directory for the motion correction is created named ``<input_name>_mocodir`` where all the files will be saved.
    2. The k-space data is separated into interleaves using ``riesling split`` and saved in the ``interleaves`` folder. The number of interleaves is controlled by the number of spokes per interleave (option ``-n``) and the length of the sliding window (``--step``).
    3. Sensitivity maps are reconstructed using ``riesling sense`` from the low-res spokes.
    4. Navigators are reconstructed using conjugate gradient SENSE, using ``riesling cg``, and saved in ``navigators``. This loop is run in parallel, number of recon jobs per iteration can be controlled by ``--batchries``.
    5. Navigators are registered to the first navigator which is used as the reference. Registration is performed with ``pymerlin reg``, again in parallel. Results from the registration are saved as a pickle file in ``registrations``. 
    6. The registration objects are merged together to a single pickle ``all_reg_param.p`` using ``pymerlin merge``.
    7. Motion correction is applied to the input data using ``pymerlin moco`` and saved to ``<input_name>_moco.h5``. 
