# Self-correction1-PDE

The solver uses two neural operators (FNO-CNN or U-Net): \\
**model0**: produces an initial prediction from the forcing field `f` \\
**model1**: iteratively corrects the prediction using the PDE residual as feedback

At each correction step:
$$u^{(t+1)} = u^{(t)} + \alpha \cdot \text{model1}([f,\ u^{(t)},\ \mathcal{R}(u^{(t)})])$$
where $\mathcal{R}(u)$ is the PDE residual.

## Helmoholtz Equation
$$\nabla^2 u + k^2 u = f, \quad (x, y) \in [0,1]^2, \quad u = 0 \text{ on } \partial\Omega$$


- 
