Getting Started
=================

pyMERLIN is a toolbox for doing motion correction of MERLIN acquisitions. It works together with [RIESLING](https://github.com/spinicist/riesling). The main steps in the motion correction framework includes:

1. Divide the complete radial dataset into interleaves (RIESLING)
2. Reconstruct each interleave into low-resolution navigator images (RIESLING)
3. Register each navigator to a target navigator, typically the first one (pyMERLIN)
4. Correct the k-space data for each interleave and combine to a motion-corrected dataset (pyMERLIN)
5. Reconstruct the final motion corrected image (RIESLING)

pyMERLIN is intended to make this into a fully-fledged integrated framework. The pipeline to do all five steps at once is not yet implemented.
