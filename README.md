# Self-correction PDE

The solver uses two neural operators (CNN-FNO-CNN or U-Net): 

**model0**: produces an initial prediction from the forcing field `f` 

**model1**: iteratively corrects the prediction using the PDE residual as feedback

At each correction step:
$$u^{(t+1)} = u^{(t)} + \alpha \cdot \text{model1}([f,\ u^{(t)},\ \mathcal{R}(u^{(t)})])$$
where $\mathcal{R}(u)$ is the PDE residual.

## Complete Dataset

## Installation

```bash
conda env create -f environment.yml
conda activate helmholtz
```

## Helmoholtz Equation
$$\nabla^2 u + k^2 u = f, \quad (x, y) \in [0,1]^2, \quad u = 0 \text{ on } \partial\Omega$$

### Training
```bash
cd Self-correction1-PDE/
python -m helmholtz_solver.train --config helmholtz_solver/configs/helmholtz.yaml
```
### Evaluation
### Without Extrapolation (k = 1, same as training)

Set `inference.k` to match the training wavenumber, then run:

```bash
python evaluate.py --config helmholtz_solver/configs/helmholtz.yaml
```

Edit `configs/helmholtz.yaml`:

```yaml
data:
  test_path: helmholtz_solver/data/helmholtz_64_test_1.mat   # test set with k=1

inference:
  k: 1        # same wavenumber as training
  T_test: 20
  step_size: 0.1
```

### With Extrapolation (k = 4, unseen at training)
Change `inference.k` and the test data path to the out-of-distribution set:

```yaml
data:
  test_path: helmholtz_solver/data/helmholtz_64_test_4.mat   # test set with k=4

inference:
  k: 4        # larger wavenumber, not seen during training
  T_test: 120  # need larger iterations
  step_size: 0.07
```

Then run:

```bash
python -m helmholtz_solver.evaluate --config helmholtz_solver/configs/helmholtz.yaml
```
