# DVS Gesture data attribution

These unlabeled inputs are derived from IBM Research's DVS128 Gesture Dataset,
introduced by Arnon Amir et al., "A Low Power, Fully Event-Based Gesture
Recognition System," CVPR 2017.

Dataset and publication:
https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system

The source dataset is distributed under the Creative Commons Attribution 4.0
International License: https://creativecommons.org/licenses/by/4.0/
Dataset attribution is to IBM Research. This license is distinct from the
repository's software license.

PINES spatially bins the 128x128 events to 32x32, keeps two polarities, and
extracts four 1.5-second windows with 60 binary time bins per window. This package
contains 2 repair-calibration development recordings, 104 semantic-audit
recordings and 160 disjoint canary recordings. Labels and original filenames
that encode labels are omitted. Dataset row indices are retained for later
evaluation. No new data were collected for this export.
