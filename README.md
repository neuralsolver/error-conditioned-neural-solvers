# Feed-Forward ENS

The solver uses two neural operators (CNN-FNO-CNN or U-Net): 

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
pip install -r HZ_solver/requirements.txt

### Evaluation

### Training
```bash
cd Self-correction1-PDE/
python -m helmholtz_solver.train --config helmholtz_solver/configs/helmholtz.yaml
```
