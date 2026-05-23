"""cifake-detection shared library.

* eval_harness   : shared metrics + JSON schema for every model
* perturbations  : BATTERY dict (JPEG/blur/noise/rescale) for robustness eval
* ensemble       : team-level ensemble + leave-one-out
* freq_detector  : Model 4 (spectral fingerprint)
* clip_probe     : Model 5 (frozen CLIP + linear/MLP probe)
"""
