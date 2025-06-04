import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxley as jx
from jaxley.channels import HH


# case 1
cell = jx.Cell([jx.Branch(ncomp=3)], parents=[-1])
cell.insert(HH())
cell.record("v")

delta_t = 0.025
current = jx.step_current(20, 10, 0.05, delta_t, 50)
cell.branch(0).loc(0.0).stimulate(current)
cell.init_states(delta_t)

v = jx.integrate(cell, delta_t=delta_t)

plt.plot(jnp.arange(0, v.shape[1])*delta_t, v.T)
plt.title(f"jaxley version {jx.__version__}")
plt.savefig("testcase1.png")

# case 2
cell = jx.read_swc("morph.swc", ncomp=1)


fig, ax = plt.subplots(1, 1, figsize=(4, 2))
_ = ax.hist(cell.nodes.length, bins=30, range=(0, 100))
_ = ax.set_title(f"v{jx.__version__}, length {sum(cell.nodes.length):.3f}")

plt.savefig("testcase2.png")

with open("out.txt", "w") as f:
    f.write(f"jaxley version {jx.__version__}\n")
    f.write(f"{v}\n")
    f.write(f"length {sum(cell.nodes.length):.3f}\n")