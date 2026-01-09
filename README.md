# HAWKSight: Hierarchical Attention-driven Weighted Kernel Network for Aerial Salient Object Detection

Abstract—
Aerial salient object detection faces unique chal-
lenges including extreme scale variations, cluttered back-
grounds, and arbitrary object orientations, leading to the
small-target dilemma and background ambiguity. To address
these issues, we propose HAWKSight: a Hierarchical Attention-
driven Weighted Kernel network. HAWKSight integrates three
key innovations: First, a multi-scale representation scheme
combining a ConvNeXt-Tiny backbone with hierarchical fea-
ture compression and an Enhanced Pyramid Pooling Module
that extends beyond standard PSPNet by incorporating full-
resolution residual connections, preserving detail from tiny
objects to large structures while maintaining boundary in-
tegrity. Second, a Dual-Attention Swin Transformer Fusion
that introduces a structured guidance-refinement paradigm,
combining shifted and regular-window attention in a novel
chained configuration to capture global context efficiently while
enabling cross-scale feature enrichment. Third, a boundary-
sensitive U-Net decoder with Squeeze-and-Excitation blocks
and a Gaussian-weighted contour loss for precise edge delin-
eation. Evaluated on ORS-4199, EORSSD, and ORSSD bench-
marks, HAWKSight achieves state-of-the-art performance with
S-measure scores of 0.8885, 0.9203, and 0.9345 respectively,
while maintaining real-time inference speeds of 105.87, 92.70,
and 98.05 FPS. Our model demonstrates exceptional boundary
precision (E-measure: 0.9451, 0.9611, 0.9751) and significantly
reduces mean absolute error compared to existing methods,
effectively resolving the speed-accuracy trade-off in aerial SOD
applications.


Date of Publication: 01 December 2025
DOI: 10.1109/JSTARS.2025.3638685
