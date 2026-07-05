# Parametric Curve Optimization Using PyTorch Gradient-Based Optimization

## Problem Statement

Solve for missing parameters, $\theta$, $M$ and $X$ in a parametric curve equation:

$$x(t) = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X$$
$$y(t) = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta)$$

Assume that we have 1500 data values for $t$ in the interval [6, 60] and given parameters such that:

- $0^\circ < \theta < 50^\circ$
- $-0.05 < M < 0.05$
- $0 < X < 100$

In other words, they gave us the points (x, y) lying on a curve for 6 < t < 60 and we need to determine three unknowns in the curve and sketch it.

## Final Solution

$$\left(t\cos(0.523599) - e^{0.030000\left|t\right|}\sin(0.3t)\sin(0.523599) + 55.000000, 42 + t\sin(0.523599) + e^{0.030000\left|t\right|}\sin(0.3t)\cos(0.523599)\right)$$

**Optimized Parameters:**

- $\theta = 30.000000^\circ$ ~ approximately ($0.523599$ radians)
- $M = 0.030000$
- $X = 55.000000$
- **Real L1 Distance** = $0.00000316$ (Mean L1 coordinate distance)

---

## Core Idea: Equations as Neurons

The main idea of this project is to use a deep learning framework (PyTorch) for numerical optimization. The parametric equations of the curve are used as our model architecture, rather than the standard layers of a neural network. The unknown variables ( $\theta, M, X$ ) are called trainable weights and they are optimized by means of the gradient descent and quasi-Newton optimization algorithms.

---

## The First Approach (`code.py`)

The first model in code.py would try to solve the equations directly:

We set $\theta$, $M$ and $X as parameters.
To approximate the curve we created a uniform set of t values with linspace(6, 60).
We optimized the parameters with the `Adam` optimizer to minimize the L1 distance, as follows: 3.

Limitation: This method produced an L1 loss of ~25.0. The optimizer was caught in a bad local minimum because we assumed that the $t$ values that are represented in the CSV were uniformly spaced and aligned, which is not true.

The following is a fit visualisation from this first approach. The curve that is expected can be seen is misaligned, and is stuck in a local minimum:

![Fitted Curve - Adam (First Approach)](code_fit_plot.png)

---

## The Mathematical Solution: De-Rotating the Curve

To solve the alignment issue of $t$, we analyzed the geometry of the curve equations.

### 1. Geometry of the Curve

Look at the given parametric equations:
$$x = t \cos\theta - e^{M|t|} \sin(0.3t) \sin\theta + X$$
$$y = 42 + t \sin\theta + e^{M|t|} \sin(0.3t) \cos\theta$$

Subtracting the offsets ($X$ and $42$) centers the curve at the origin:
$$x - X = t \cos\theta - e^{M|t|} \sin(0.3t) \sin\theta$$
$$y - 42 = t \sin\theta + e^{M|t|} \sin(0.3t) \cos\theta$$

Now, we replace the complicated exponential/sine term with a simple variable $A$:
$$A = e^{M|t|} \sin(0.3t)$$

This simplifies the equations to:
$$x - X = t \cos\theta - A \sin\theta$$
$$y - 42 = t \sin\theta + A \cos\theta$$

### 2. Identifying the Rotation Matrix

This pattern perfectly matches the standard 2D rotation equations.
When any coordinate $(a, b)$ is rotated by an angle $\theta$, the rotated coordinates $(x', y')$ are given by:
$$x' = a \cos\theta - b \sin\theta$$
$$y' = a \sin\theta + b \cos\theta$$

By comparing the two sets of equations:

- $a = t$
- $b = A = e^{M|t|} \sin(0.3t)$

This reveals that the original coordinate before rotation was $(t, A)$!

The system can be written in matrix form as:

$$
\begin{bmatrix} x - X \\\\ y - 42 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} t \\\\ A \end{bmatrix}
$$

The matrix below is a standard rotation matrix:

$$
\begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix}
$$

### 3. Reverse Rotation (De-Rotation)

To map each individual point in `xy_data.csv` to its correct value of $t$, we can apply the inverse rotation (rotating by $-\theta$). The inverse rotation matrix is:

$$
\begin{bmatrix} t \\\\ z_{actual} \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\\\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x - X \\\\ y - 42 \end{bmatrix}
$$

This simplifies to:

$$
\begin{aligned}
t &= (x - X)\cos\theta + (y - 42)\sin\theta \\\\
z_{actual} &= -(x - X)\sin\theta + (y - 42)\cos\theta
\end{aligned}
$$

Using this method, the exact parameter $t$ for each individual coordinate $(x, y)$ in the CSV is dynamically calculated. We then compare our model $z_{model} = e^{M|t|} \sin(0.3t)$ against $z_{actual}$.

---

## Second Approach: L-BFGS & Bounded Constraints (`optimize.py`)

Using this mathematical rotation trick, we implemented `optimize.py` with the following enhancements:

1. **L-BFGS Optimizer**: We used PyTorch's L-BFGS optimizer to quickly find the minimum.
2. **Bounds Clamping inside Closure**: Since L-BFGS performs line searches and evaluates the closure multiple times per step, we moved parameter clamping inside the `closure()` loop. This prevents parameters from stepping out of bounds during intermediate optimization steps.
3. **Loss Computation**: We optimize in the rotated 1D space, and then output the true 2D L1 loss ($|x_{pred}-x_{csv}| + |y_{pred}-y_{csv}|$) at the end.

This approach successfully avoids local minima, converging to:

- **$\theta$ (Theta)**: $30.0000^\circ$ (which is $\frac{\pi}{6}$ radians)
- **$M$**: $0.03000$
- **$X$**: $55.0000$
- **Real L1 Loss**: **$0.00000316$** (effectively $0$)

Here is the final optimized curve fit plotted against the raw data points showing perfect convergence:

![Fitted Curve - L-BFGS (Second Approach)](fit_plot.png)

---

## Repository Structure

The workspace is structured as follows:

```text
AI_rd/
├── optimize.py          # Core implementation (PyTorch L-BFGS, rotation trick)
├── code.py              # First approach (PyTorch Adam with linspace)
├── plot_data.py         # Utility script to plot raw data points
├── xy_data.csv          # Input data (1500 points)
├── results.txt          # Final grading parameters and score
├── requirements.txt     # Python package requirements
├── fit_plot.png         # Final curve fit image (Second Approach)
├── code_fit_plot.png    # Misaligned fit image (First Approach)
└── README.md            # Project documentation
```

---

## Technical Analysis

### 1. Optimization Algorithm Comparison

Adam (First-order Stochastic Gradient Descent): Adam updates parameters using moving averages of the gradients. The parametric equations (high frequency oscillations from $\sin(0.3t)$ and scaling from $e^{M|t|}$) used in code.py are not convex, so the optimizer stuck in local minima (L1 loss $\approx 25.0$).
L-BFGS uses an approximation to the inverse Hessian matrix of second derivatives in order to select search directions. L-BFGS was significantly more efficient in `optimize.py`, converged to the global minimum, and performed better. It uses its line search function to adaptively pick steps along the curve, thus allowing for high precision parameter estimation.

### 2. Clamping in the Closure – inside.

One of the major technical challenges addressed was the enforcement of constraints. In PyTorch's L-BFGS implementation, the optimizer will internally do line search on the model, and call the `closure()` block several times per step.
If the parameters are clamped outside of the closure loop:
The 1. L-BFGS steps parameters out of their bounds error occurred. 2. The closure is a way to compute gradients at out-of-bounds coordinates. 3. The optimizer becomes confused, takes ‘short cuts’ and gets off the path.

The move of the .clamp\_() calls into the closure() loop guarantees that L-BFGS always accesses states that comply with the bounds and are consistent with each other, leading to stable and consistent convergence.

### 3. Evaluating the loss of the target area and estimating the dimensional loss.

The L1 loss is defined in the assignment as:

$$
\text{L1 Distance} = \text{mean}(|x_{\text{pred}} - x_{\text{csv}}| + |y_{\text{pred}} - y_{\text{csv}}|)
$$

Optimizing in the rotated coordinate space $z_{\text{actual}}$ is possible in our L-BFGS guide loop because we know that the point where $z_{\text{actual}}$ is optimized is the same as where $z_{\text{model}} = 0$. Geometrically, the loss for the 2D l1 loss is scaled by:

$$
\text{L1}_{\text{2D}} = |z_{\text{model}} - z_{\text{actual}}| \cdot (|\sin\theta| + |\cos\theta|)
$$

After all the computations are done, the distance between the cursor and the real x, y position is computed and logged into the standard (x, y) space, ensuring that the final evaluation exactly matches the scoring criteria of the assignment and is not based on a scaling bias.

---

## References & Citations

1. **L-BFGS & Numerical Optimization**:
   - Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). _A limited memory algorithm for bound constrained optimization_. _SIAM Journal on Scientific Computing, 16_(5), 1190–1208. https://doi.org/10.1137/0916069
   - Nocedal, J., & Wright, S. J. (2006). _Numerical optimization_ (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5

2. **Adam Optimizer**:
   - Kingma, D. P., & Ba, J. (2015). _Adam: A method for stochastic optimization_. _International Conference on Learning Representations (ICLR)_. https://arxiv.org/abs/1412.6980

3. **PyTorch Framework**:
   - Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... Chintala, S. (2019). _PyTorch: An imperative style, high-performance deep learning library_. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, & R. Garnett (Eds.), _Advances in Neural Information Processing Systems_ (Vol. 32). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
