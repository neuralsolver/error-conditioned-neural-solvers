# Feed-Forward ENS

The solver uses two neural operators (FNO): 

**model0**: produces an initial prediction from the forcing field `f`, for Helmoholtz and Poisson Equation, or intial condition `u_0` for Burgers Equation, Navier Stoke, and Komolgorov flow.

**model1**: iteratively corrects the prediction using the PDE residual as feedback

At each correction step:
$$u^{(t+1)} = u^{(t)} + \alpha \cdot \text{model1}([f,\ u^{(t)},\ \mathcal{R}(u^{(t)})])$$
where $\mathcal{R}(u)$ is the PDE residual.

## Complete Dataset
## All Pretrained models
## Installation

```bash
conda env create -f environment.yml
conda activate helmholtz
```

## Helmoholtz Equation
$$\nabla^2 u + k^2 u + \lambda u^3 = f, \quad (x, y) \in [0,1]^2, \quad u = 0 \text{ on } \partial\Omega$$ 

For linear Helmholtz equation, we use $k=1$, $\lambda=0$ with $128 \times 128$ resolution for training. 

For nonlinear Helmholtz equation, we use $k=2$, $\lambda=1$ with $128 \times 128$ resoltion for training.

### Setup
```bash
pip install -r HZ_solver/requirements.txt
```

### Training
Train from scratch:

```bash
cd HZ_solver
python -u train.py --config config/hz.json
```

### Evaluation
#### Indistribution:
Use a Helmholtz test set with the same equation setting as training.

```bash
python -u evaluate.py --config config/hz.json
```
#### Extrapolation:
Use a Helmholtz test set generated with a different wave number.

Example for $k=3$:

```bash
python -u evaluate.py --config config/hz_extrapolation.json
```

#### Super-resolution:
Use a higher-resolution Helmholtz test set.

Example for $256 \times 256$:

```bash
python -u evaluate.py --config config/hz_superresolution.json
```

#### Cross-equation:
Zero-shot test on Poisson equation:

```bash
python -u evaluate.py --config config/hz_crossequation.json
```





