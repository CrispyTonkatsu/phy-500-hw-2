# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% jupyter={"source_hidden": true}
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from IPython.display import Latex, Markdown, display_latex, display_markdown

# +
# %matplotlib inline
# -

# %% [markdown]
r"""
# First Order Systems
## Autonomous Systems
$$
\dot{y} = y(y - 1)(2 - y),\ y \geq 0
$$

### Find all equilibrium concentrations
$$
\dot{y} = 0 \rightarrow y = 0, 1, 2
$$

### Plot the phase line on the $(y,\dot{y})$-plane
"""

# %% jupyter={"source_hidden": true}
plt.figure(figsize=(8, 4))
plt.title("Phase line graph")


def fn1(y):
    return y * (y - 1) * (2 - y)


y = np.linspace(-0.2, 2.2, 100)
dy = fn1(y)

plt.plot(y, dy, zorder=1)

eq_y = np.array([0, 1, 2])
eq_dy = fn1(eq_y)

plt.scatter(eq_y, eq_dy, color="red", zorder=3)

step = 5
y = y[::step]
dy = dy[::step]
ddy = -3 * y**2 + 6 * y - 2

flow_dir = np.sign(dy)
u = flow_dir
v = ddy * flow_dir

plt.quiver(y, dy, u, v, zorder=2)

plt.xlabel("y")
plt.ylabel("dy")
plt.grid(True)
plt.show()

# %% [markdown]
r"""
### Classify each equilibrium as a source or sink based on the direction of the arrows on your phase line
$$
y = 0 \rightarrow \text{sink/stable}
$$

$$
y = 1 \rightarrow \text{source/unstable}
$$

$$
y = 2 \rightarrow \text{sink/stable}
$$

### Based on your phase line, explain briefly why this system exhibits bistable behavior
This is a bistable system given there are two states that are considered stable for the system.
One can think of this as a light switch where it'll either be on or off.
It could get stuck in the middle, but any small disturbance will take it back to either of the two.
"""

# %% [markdown]
r"""
## Non-Autonomous Systems
$$
\dot{y} = -ty
$$

### Use the classic "separate and integrate" technique to obtain the general solution to the ODE
$$
\begin{align}
    \frac{dy}{dt} &= -ty \\
    \int y^{-1} dy &= -\int t \,dt \\
    \ln(y) &= -\frac{1}{2} t^{2} + C\\
    y &= Ce^{-\frac{1}{2} t^{2}}
\end{align}
$$

### Plot a slope field on the $(t, y)$-plane
"""


# %% jupyter={"source_hidden": true}
plt.figure(figsize=(8, 4))


def diff_fn2(t, y):
    return -t * y


def num_solve(t_range, y_init, h):
    output = np.zeros_like(t_range)
    output[0] = y_init

    for i in range(len(t_range) - 1):
        t = t_range[i]
        y = output[i]
        output[i + 1] = y + h * diff_fn2(t, y)

    return output


def fn2(t, C):
    return C * np.exp(-(1 / 2) * t**2)


t = np.linspace(0, 5)
y = np.linspace(0, 5)

h = t[1] - t[0]
plt.title(f"Capacitor Discharge with $h = {h}$")

y_1 = fn2(t, 1)
plt.plot(t, y_1, color="blue", label="$y_0 = 1$")
plt.plot(t, num_solve(t, 1, h), "bo", label="numerical $y_0 = 1$")

y_2 = fn2(t, 2)
plt.plot(t, y_2, "red", label="$y_0 = 2$")
plt.plot(t, num_solve(t, 2, h), "ro", label="numerical $y_0 = 2$")

y_3 = fn2(t, 3)
plt.plot(t, y_3, color="green", label="$y_0 = 3$")
plt.plot(t, num_solve(t, 3, h), "go", label="numerical $y_0 = 3$")

step = 2
T, Y = np.meshgrid(t[::step], y[::step])

dT = np.ones_like(T)
dY = diff_fn2(T, Y)
M = np.hypot(dT, dY)

arrow_length = 0.2
dT = arrow_length * (dT / M)
dY = arrow_length * (dY / M)

plt.quiver(T, Y, dT, dY, dT)

plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()

# %% [markdown]
r"""
# Linear Dynamical Systems
## Equilibrium Points
Since these systems are homogeneous linear, the equilibrium
$Ax = \vec{0}$ can only have solution $x = \vec{0}$.
This is because all 6 matrices have $\det(A) \neq 0$.
"""

# %% jupyter={"source_hidden": true}
# System matrices
A1 = sp.Matrix([[0, 1], [-2, -3]])
A2 = sp.Matrix([[0, 1], [-2, 3]])
A3 = sp.Matrix([[0, 1], [-5, -2]])
A4 = sp.Matrix([[0, 1], [-5, 2]])
A5 = sp.Matrix([[0, 1], [-1, 0]])
A6 = sp.Matrix([[0, 1], [1, 0]])

matrices = [A1, A2, A3, A4, A5, A6]

for i, mat in enumerate(matrices):
    det = sp.det(mat)
    display_latex(Latex(f"$\\det(A_{i + 1}) = {det}$"))

# %% [markdown]
r"""
## Eigen Values, Eigen Vectors and Classification
"""

# %% jupyter={"source_hidden": true}


def classify(eigen_vals):
    if len(eigen_vals["real"]) == 2:
        a, b = eigen_vals["real"]

        if a < 0 and b < 0:
            return "Stable Node"

        elif a > 0 and b > 0:
            return "Unstable Node"

        elif sp.sign(a) != sp.sign(b):
            return "Saddle"

        return "None"

    elif len(eigen_vals["complex"]) == 2:
        a, b = eigen_vals["complex"]

        if sp.conjugate(a) != b:
            return "None"

        if sp.re(a) < 0 and sp.re(b) < 0:
            return "Stable Spiral"

        elif sp.re(a) > 0 and sp.re(b) > 0:
            return "Unstable Spiral"

        elif sp.re(a) == 0 and sp.re(b) == 0:
            return "Center"

        return "None"

    else:
        return "None"


for i, mat in enumerate(matrices):
    eigen_vals = {"real": [], "complex": []}

    latex_out = f"$$A_{i + 1} \\rightarrow"

    eigen_vecs = mat.eigenvects()
    for j, (eigval, mult, eigvect) in enumerate(eigen_vecs):
        latex_out += f"\\lambda_{j} = {sp.latex(eigval)}, \\ "
        latex_out += f"\\vec{{\\lambda_{j}}} = {sp.latex(eigvect[0])} \\ "

        if eigval.is_real:
            eigen_vals["real"].append(eigval)
        else:
            eigen_vals["complex"].append(eigval)

    latex_out += "\\rightarrow \\text{{" + classify(eigen_vals) + "}}$$"

    display_latex(Latex(latex_out))

# %% [markdown]
r"""
## Pointcare Trace Diagram Interpretation
"""

# %% jupyter={"source_hidden": true}
plt.figure(figsize=(8, 4))
plt.title("Pointcare Trace Diagram")
plt.grid(True)

trace = np.linspace(-6, 6)
det_curve = trace**2 / 4

plt.plot(trace, det_curve, label="Pointcare Curve")

for i, mat in enumerate(matrices):
    mat_trace = mat.trace()
    mat_det = float(mat.det().evalf())  # pyright: ignore[reportAttributeAccessIssue]

    plt.scatter(mat_trace, mat_det, label=f"$A_{{{i + 1}}}$")

plt.xlabel(r"trace")
plt.ylabel(r"det")
plt.legend()
plt.show()

# %% [markdown]
r"""
### $A_1$
For this system, it can be seen in both eigen values ($a,b \in R$ \& a,b < 0) that the system is converging towards stability at the origin.
Similarly, the location of the trace and determinant of the matrix, it is considered a sink.
This means that all trajectories of that flavor will converge at the origin over the long term.

In a more physical interpretation, the spring mass will eventually remain still without the need to oscilate (unlike other systems I'll discuss later).
This would be more commonly described as an overdamped system.

### $A_2$
Unlike $A_1$, this system will never converge to the equilibrium point. This can be seen in the source node categorization.

In the context of our spring mass, this means that the spring mass will be only stable if the state is equilibrium, but the moment it is disturbed it will move further from it over time.
This would be more commonly described as exponential growth.

### $A_3$
This would be considered an underdamped system.
The core features here can be seen in the spiral sink categorization and the negative real parts of the eigenvalues.
This means that it will eventually converge at the origin, but the oscillatory nature of the eigenvalues help characterize the fact that it will oscillate as it approaches equilibrium.

### $A_4$
Like $A_2$, this is exponential growht.
However, the key distinction is it's similarity with $A_3$ in that it will oscillate as it exponentially grows in magnitude.

### $A_5$
This would be an undamped system.
Given there are no real parts in the eigen values, it will continue to oscillate as no energy is being removed or added from the system.

### $A_6$
This system would be seen as a semi-stable system.
This is due to the system not exponentially growing but it having a state where it converges to the origin while having another one where it grows away from the origin.
This would be the case of a spring that pushes outwards instead of inwards, therefore whenever it can achieve equilibrium it gets pushed away.
"""

# %% [markdown]
r"""
## Analytical Solutions using the Matrix Exponential

$$
\dot{x} = Ax \rightarrow x(t) = e^{At}x_0
$$

$$
x_0 = \begin{bmatrix}
q_0 \\ v_0
\end{bmatrix}
$$
"""

# %% jupyter={"source_hidden": true}
for i, mat in enumerate(matrices):
    P, D = mat.diagonalize()
    P_inv = P.inv()

    display_markdown(Markdown(f"### System $A{i + 1}$"))
    display_latex(
        Latex(f"""$$
        x(t) = {sp.latex(P)}e^{{{sp.latex(D)}t}}{sp.latex(P_inv)} x_0
        $$""")
    )

    t = sp.symbols("t", real=True)
    q0, v0 = sp.symbols("q_0 v_0", real=True)
    x0 = sp.Matrix([q0, v0])

    exp_Dt = D.applyfunc(lambda entry, t=t: sp.exp(entry * t))
    exp_At = P * exp_Dt * P_inv
    x_t = sp.simplify(exp_At * x0)

    display_latex(
        Latex(f"""$$
        x(t) = {sp.latex(x_t)}
        $$""")
    )

# %% [markdown]
r"""
## Vector fields, Phase portraits and Eigen vector directions
"""

# %% jupyter={"source_hidden": true}


def matrix_slope_field(A):
    x = np.linspace(-3, 3)
    y = np.linspace(-3, 3)

    step = 2
    X, Y = np.meshgrid(x[::step], y[::step])
    U = A[0, 0] * X + A[0, 1] * Y
    V = A[1, 0] * X + A[1, 1] * Y

    M = np.sqrt(U**2 + V**2)
    M[M == 0] = 1.0

    U_unit = U / M
    V_unit = V / M

    plt.figure(figsize=(8, 4))
    plt.title(f"Slope field for system $A_{1}$")

    plt.quiver(X, Y, U_unit, V_unit, M)

    plt.xlabel("q")
    plt.ylabel("v")
    plt.grid(True)
    plt.show()


matrix_slope_field(sp.matrix2numpy(A1, dtype=float))
