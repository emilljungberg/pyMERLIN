Registration Framework
=========================

.. automodule::
    pymerlin.reg

Core registration Functions
-----------------------------
The main workhorse of the registration is the `ants_pyramid` function. The registration is written in ITK but modeled on the settings in the ANTs pyramid registration, and thus the name.

.. autosummary::
    :toctree: generated
    :nosignatures:

    pymerlin.reg.ants_pyramid

Filter and Masking
-----------------------------
Some tools for masking and thresholding

.. autosummary::
    :toctree: generated
    :nosignatures:

    pymerlin.reg.sphere_mask
    pymerlin.reg.brain_mask
    pymerlin.reg.otsu_filter
    pymerlin.reg.winsorize_image
    pymerlin.reg.threshold_image
    pymerlin.reg.histogram_threshold_estimator

Registration help
-----------------------------
Tools for the registration framework

.. autosummary::
    :toctree: generated
    :nosignatures:

    pymerlin.reg.versor_reg_summary
    pymerlin.reg.versor_watcher
    pymerlin.reg.resample_image
    pymerlin.reg.get_versor_factors
    pymerlin.reg.setup_optimizer
    pymerlin.reg.versor_resample
    pymerlin.reg.make_opt_par
