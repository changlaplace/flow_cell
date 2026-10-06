# Coherent Two-Metasurface Flow Cytometry Prototype

This folder contains a standalone Python implementation of the Jupyter notebook for the proposal-aligned coherent flow-cytometry model.

## Files

- `flow_cytometry_end_to_end.py` — stable command-line entry point.
- `flow_cytometry/config.py` — optical, data, and optimization configuration.
- `flow_cytometry/optics.py` — ASM propagation and phase-only optical components.
- `flow_cytometry/data.py` — synthetic cell generation and dataset.
- `flow_cytometry/model.py` — combined optical/digital classifier.
- `flow_cytometry/metrics.py` — Bhattacharyya separability metrics.
- `flow_cytometry/engine.py` — evaluation and training loops.
- `flow_cytometry/plotting.py` — saved visualizations.
- `flow_cytometry/utils.py` — shared output and reproducibility helpers.
- This README — setup and usage instructions.

The original command-line interface is unchanged after this modular split.

## Optical model

The model is:

```text
coherent visible laser
        ↓
trainable phase mask M1
        ↓ ASM
thin phase-object cell
        ↓ ASM
trainable phase mask M2
        ↓ ASM
low-resolution intensity detector
        ↓
lightweight digital classifier
```

Multiple copies of this paired optical transform can be placed along the flow direction to model the proposal's 1D array of optical stations.

The current cell model is intentionally simple:

```text
t_cell(x,y) = exp(i * phi_cell(x,y))
```

So the cell is a 2D thin phase object. There is no 3D scattering model yet.

## Parameters

Proposal-aligned assumptions:

- coherent visible illumination
- cell flow speed: `1 m/s`
- target architecture: roughly `10–100` optical stations
- phase-only metasurfaces
- Angular Spectrum Method propagation
- compressed low-resolution detector readout
- lightweight digital classifier

Numerical defaults borrowed from the supplied Behrens/Li/Monticone paper:

- wavelength: `632.8 nm`
- sampling pitch: `0.35 µm`
- paper reference grid: `1200 x 1200`
- paper reference aperture: `420 µm`
- propagation scale: `16 mm + 4 mm`

Because the proposal uses two metasurfaces around the cell, the script currently adapts the propagation distance to:

```text
M1
↓ 8 mm
cell
↓ 8 mm
M2
↓ 4 mm
sensor
```

This is a modeling default, not a proposal-fixed geometry.

### Detector sampling

For the default `192 x 192` grid with `0.35 µm` sampling and a `4 x 4`
detector, each pooled detector pixel covers `48 x 48` simulation samples.
The resulting numerical effective pixel pitch is therefore:

```text
(192 / 4) x 0.35 µm = 16.8 µm
```

The modeled active detector area is `67.2 x 67.2 µm²`, with gapless average
pooling (an idealized 100% fill factor). A physical detector pixel pitch can
differ if relay magnification is introduced. At the paper-reference
`420 µm` aperture, a direct `4 x 4` partition would instead correspond to
`105 µm` per detector pixel.

At the assumed `1 m/s` flow speed, a cell crosses the modeled `67.2 µm`
active width in `67.2 µs`. This gives two useful detector timing targets:

```text
minimum for >=1 frame per active-area transit: 1 / 67.2 µs = 14.9 kfps
one-pixel motion limit: exposure <= 16.8 µs; continuous sampling >= 59.5 kfps
```

The required triggered event rate may be lower and depends on cell spacing.
These timing values are engineering targets derived from the current geometry,
not measured hardware specifications.

## Install

Recommended Python: 3.10+

```bash
pip install torch numpy matplotlib
```

For an NVIDIA GPU, install the correct CUDA-enabled PyTorch build from the official PyTorch instructions.

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Quick run

This uses the lightweight default:

- grid: `192 x 192`
- stations: `3`
- detector: `4 x 4`
- no training

```bash
python flow_cytometry_end_to_end.py
```

Outputs are saved to:

```text
flow_cytometry_outputs/
```

Typical files:

```text
config.json
optical_setup.png
toy_phase_objects.png
forward_sanity.png
phase_masks.png
feature_correlation.png
metrics_untrained.json
```

## Run end-to-end training

```bash
python flow_cytometry_end_to_end.py --run-training
```

Additional outputs:

```text
training_curves.png
metrics.json
model.pt
```

The training jointly optimizes:

```text
M1 phase
+
M2 phase
+
digital classifier
```

using cross-entropy.

The script also reports the multivariate joint Bhattacharyya distance of the compressed detector representation.

## A6000 run

A more serious run can use:

```bash
python flow_cytometry_end_to_end.py \
    --run-training \
    --grid-N 1200 \
    --stations 10 \
    --sensor-N 4 \
    --epochs 50 \
    --output-dir results_a6000
```

`1200 x 1200` with multiple ASM propagations and many stations can consume substantial GPU memory and time.

If memory is too high, first try:

```bash
--grid-N 512
```

or:

```bash
--stations 4
```

and then scale up.

## Change backend

Linear classifier:

```bash
--backend linear
```

Small MLP:

```bash
--backend mlp
```

The linear backend is useful when you want the optical frontend to carry as much of the feature extraction burden as possible.

## Saved-output behavior

Unlike the notebook, the Python script saves figures and metrics automatically instead of relying on notebook cell output.

Every run saves:

```text
optical setup
phase-cell examples
forward-propagation sanity figure
current phase masks
feature-correlation matrix
configuration
metrics
```

Training runs additionally save:

```text
training curves
trained weights
final metrics/history
```

## Recommended first scientific comparison

The next useful ablation is:

```text
1. fixed / unoptimized optics
2. M2 trainable only
3. M1 + M2 trainable
```

using the same:

```text
dataset
sensor bottleneck
digital backend
training schedule
```

Then compare:

```text
classification accuracy
joint Bhattacharyya distance
detector-feature correlation
```

Additional sweeps:

```text
sensor_N = 1, 2, 4, 8, 10
stations = 1, 2, 4, 10
coherent vs phase-randomized/incoherent control
linear vs MLP backend
```

## Replacing the synthetic cells

The current toy dataset only exists so that the full optical-learning pipeline can run without external biological data. Its four deliberately separable proof-of-concept classes are a small round cell, a large elongated cell, a bilobed cell, and an annular/granular cell. These synthetic labels should not be interpreted as biological cell types or as evidence of real-world accuracy.

For measured phase data, the dataset only needs to return:

```python
phi_cell, label
```

The cell interaction remains:

```python
U = U * torch.exp(1j * phi_cell)
```

If amplitude information becomes available:

```python
U = U * amplitude * torch.exp(1j * phi_cell)
```

If you later move to a 3D refractive-index model, replace this thin-object interaction with multislice propagation, BPM, or another scattering solver. The rest of the pipeline can stay nearly unchanged.
