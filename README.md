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
* **Adam (First-order Stochastic Gradient Descent)**: Adam updates parameters based on moving averages of gradients. In `code.py`, the optimizer was trapped in local minima (L1 loss $\approx 25.0$) due to the non-convex nature of the parametric equations (specifically, high-frequency oscillations from the $\sin(0.3t)$ term and scaling from $e^{M|t|}$).
* **L-BFGS (Quasi-Newton Second-order Line Search)**: L-BFGS estimates the inverse Hessian matrix of second derivatives to choose search directions. In `optimize.py`, L-BFGS was much faster and successfully converged to the global minimum. Its line search mechanism dynamically adjusted steps along the curvature, enabling high-precision parameter estimation.

### 2. Constraint Clamping inside the Closure
A key technical issue solved was constraint enforcement. In PyTorch's L-BFGS implementation, the optimizer performs internal line searches, calling the `closure()` block multiple times per step.
If parameters are clamped *outside* the closure loop:
1. L-BFGS steps parameters outside their bounds.
2. The closure evaluates gradients at out-of-bounds coordinates.
3. The optimizer gets confused, breaks gradient trajectories, and diverges.

Moving the `.clamp_()` operations directly **inside** the `closure()` loop ensures that L-BFGS always evaluates bounds-compliant, consistent states, resulting in steady and stable convergence.

### 3. Loss Metric & Dimensional Scaling
The assignment defines the L1 loss as:
$$
\text{L1 Distance} = \text{mean}(|x_{\text{pred}} - x_{\text{csv}}| + |y_{\text{pred}} - y_{\text{csv}}|)
$$

In our L-BFGS guide loop, optimizing directly in the rotated coordinate space $z_{\text{actual}}$ works because $z_{\text{model}} - z_{\text{actual}} = 0$ at the true solution. Geometrically, the 2D L1 loss is scaled by:
$$
\text{L1}_{\text{2D}} = |z_{\text{model}} - z_{\text{actual}}| \cdot (|\sin\theta| + |\cos\theta|)
$$

By computing and logging the **Real L1 Distance** in standard $(x, y)$ space at the end of the script, we guarantee that the final evaluation precisely matches the assignment's scoring criteria without any scaling bias.

---

## References & Citations

1. **L-BFGS & Numerical Optimization**:
   * Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190-1208.
   * Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer-Verlag.
2. **Adam Optimizer**:
   * Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
3. **PyTorch Framework**:
   * Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.

