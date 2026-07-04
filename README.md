# Parametric Curve Optimization Using PyTorch Gradient-Based Optimization

## Problem Statement
Finding unknown parameters ($\theta$, $M$, $X$) in a parametric curve equation:

$$x(t) = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X$$
$$y(t) = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta)$$

Given 1500 data points for $t \in [6, 60]$ with parameter constraints:
* $0^\circ < \theta < 50^\circ$
* $-0.05 < M < 0.05$
* $0 < X < 100$

So in plain words, what was given was $(x,y)$ points along a curve with $6 < t < 60$, and we have to find out three unknowns in the curve and plot this.

## Final Solution
$$\left(t\cos(0.523599) - e^{0.030000\left|t\right|}\sin(0.3t)\sin(0.523599) + 55.000000, 42 + t\sin(0.523599) + e^{0.030000\left|t\right|}\sin(0.3t)\cos(0.523599)\right)$$

**Optimized Parameters:**
* $\theta = 30.000000^\circ$ ~ approximately ($0.523599$ radians)
* $M = 0.030000$
* $X = 55.000000$
* **Real L1 Distance** = $0.00000316$ (Mean L1 coordinate distance)

---

## Core Idea: Equations as Neurons

The central concept of this project is to leverage a deep learning framework (PyTorch) for numerical optimization. Instead of traditional neural network layers, we treat the parametric equations of the curve directly as our model architecture. The unknown variables ($\theta$, $M$, $X$) are defined as trainable weights, and we optimize them using gradient descent and quasi-Newton optimization algorithms.

---

## The First Approach (`code.py`)

In `code.py`, the initial model attempted to solve the equations directly:

1. We initialized $\theta$, $M$, and $X$ as parameters.
2. We generated a uniform grid of `t` values using `linspace(6, 60)` to approximate the curve.
3. We optimized the parameters using the `Adam` optimizer to minimize the L1 distance.

**Limitation:** This approach resulted in a high L1 loss of approximately **25.0**. The optimizer got trapped in severe local minima because we assumed the $t$ values corresponding to the CSV data points were uniformly spaced and aligned, which was not the case.

Below is the fit visualization from this first approach. You can see the predicted curve is misaligned and trapped in a local minimum:

![Fitted Curve - Adam (First Approach)](code_fit_plot.png)

---

## The Mathematical Breakthrough: De-rotating the Curve

To solve the alignment issue of $t$, we analyzed the geometry of the curve equations.

Look at the given parametric equations:
$$x = t \cos\theta - e^{M|t|} \sin(0.3t) \sin\theta + X$$
$$y = 42 + t \sin\theta + e^{M|t|} \sin(0.3t) \cos\theta$$

If we subtract the offsets ($X$ and $42$) to center the curve at the origin:
$$x - X = t \cos\theta - e^{M|t|} \sin(0.3t) \sin\theta$$
$$y - 42 = t \sin\theta + e^{M|t|} \sin(0.3t) \cos\theta$$

Now, replace the complex expression with a simple variable $A$:
$$A = e^{M|t|} \sin(0.3t)$$

This simplifies the equations to:
$$x - X = t \cos\theta - A \sin\theta$$
$$y - 42 = t \sin\theta + A \cos\theta$$

### Recognizing the Rotation Matrix

This pattern perfectly matches the standard 2D rotation equations.
If we rotate any coordinate $(a, b)$ by an angle $\theta$, the rotated coordinates $(x', y')$ are given by:
$$x' = a \cos\theta - b \sin\theta$$
$$y' = a \sin\theta + b \cos\theta$$

By comparing the two sets of equations:

- $a = t$
- $b = A = e^{M|t|} \sin(0.3t)$

This reveals that the original coordinate before rotation was $(t, A)$!
In matrix form, the system is expressed as:
$$\begin{bmatrix} x - X \\ y - 42 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} t \\ A \end{bmatrix}$$

The matrix $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ is a standard rotation matrix.

### Reverse Rotation (De-rotation)

To align each individual point in `xy_data.csv` to its correct $t$ value, we can apply the inverse rotation (rotating by $-\theta$). The inverse rotation matrix is:
$$\begin{bmatrix} t \\ z_{actual} \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x - X \\ y - 42 \end{bmatrix}$$

This simplifies to:
$$t = (x - X)\cos\theta + (y - 42)\sin\theta$$
$$z_{actual} = -(x - X)\sin\theta + (y - 42)\cos\theta$$

By doing this, **we dynamically compute the exact parameter $t$ for every single coordinate $(x, y)$ in the CSV**.
We then compare our model $z_{model} = e^{M|t|} \sin(0.3t)$ against $z_{actual}$.

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
