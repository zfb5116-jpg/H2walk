# Stairs L0b2 LiDAR Terrain Reconstruction

This package contains a trained PyTorch LiDAR terrain reconstructor and the
source code used to train it for the `stairs_l0b2` simulation terrain.

## Contents

- `lidar_reconstructor_phase1.pt`: trained model weights (step 39,999).
- `rsl_rl/networks/lidar_terrain_reconstructor.py`: reconstruction network.
- `scripts/train_yu_lidar_reconstructor.py`: training and evaluation script.
- `stairs-l0b2-recon.canvas.tsx`: Chinese-language explanation of the
  reconstruction pipeline and reported metrics.

The raw training log is intentionally omitted because it contains machine-local
paths and process environment details. The log reports 40,000 training steps
with 64 environments. The model produces a 17 x 33 height map at 5 cm cell
spacing. Its recorded final refined mean absolute errors were 3.5 mm on flat
terrain, 4.3 mm for up L0, 3.2 mm for down L0, 13.7 mm for down L4, 12.1 mm
for up L9, and 10.1 mm for down L9. The L0 down-step geometry may be smoothed
by the LiDAR hit filtering, so a low average error alone does not guarantee the
step edge was reconstructed correctly.

This package contains model weights and code, not an exported terrain mesh,
point cloud, or height-map artifact.
