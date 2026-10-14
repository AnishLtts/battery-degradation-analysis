import numpy as np, matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter

FOLDER = r"C:\Users\99017994\OneDrive - LTTS\Documents\1. BatteryAgingARC-FY08Q4"
CELL = "B0005"

m = loadmat(FOLDER + "\\" + CELL + ".mat")
cycles = m[CELL][0, 0]["cycle"][0]
print("records:", len(cycles))

vgrid = np.linspace(3.2, 4.1, 300)      # useful voltage range, 3 mV steps
curves, caps, icas = [], [], []

for c in cycles:
    if str(np.squeeze(c["type"])) != "discharge":
        continue
    d = c["data"][0, 0]
    v = np.squeeze(d["Voltage_measured"]).astype(float)
    i = np.squeeze(d["Current_measured"]).astype(float)
    t = np.squeeze(d["Time"]).astype(float)
    if v.size < 20:                     # skip broken cycles
        continue
    q = np.concatenate([[0], np.cumsum(np.abs(i[:-1]) * np.diff(t))]) / 3600

    # keep only the real discharge (no rest at start, no voltage bounce-back at end)
    k = np.argmin(v)
    on = np.abs(i[:k + 1]) > 0.1
    v, q = v[:k + 1][on], q[:k + 1][on]

    curves.append((v, q))
    caps.append(float(np.squeeze(d["Capacity"])))

    # ICA: voltage strictly decreasing -> flip -> interpolate -> dQ/dV -> light smoothing
    vm = np.minimum.accumulate(v)
    keep = np.r_[True, np.diff(vm) < 0]
    qg = np.interp(vgrid, vm[keep][::-1], q[keep][::-1])
    icas.append(savgol_filter(-np.gradient(qg, vgrid), 15, 3))

n = len(curves)
print("discharge cycles:", n)
print("capacity %.3f -> %.3f Ah  (fade %.1f%%)"
      % (caps[0], caps[-1], (1 - caps[-1] / caps[0]) * 100))

picks = np.linspace(0, n - 1, 5).astype(int)
col = plt.cm.viridis(np.linspace(0, .9, 5))
_, ax = plt.subplots(1, 3, figsize=(16, 5))

for c_, k in zip(col, picks):
    ax[0].plot(curves[k][1], curves[k][0], color=c_, label="cycle %d" % (k + 1))
    ax[1].plot(vgrid, icas[k], color=c_, label="cycle %d" % (k + 1))

ax[0].set(xlabel="Capacity (Ah)", ylabel="Voltage (V)", title="Voltage-Capacity")
ax[1].set(xlabel="Voltage (V)", ylabel="dQ/dV", title="ICA curves")
ax[2].plot(range(1, n + 1), caps, "o-", ms=3, color="crimson")
ax[2].set(xlabel="Cycle", ylabel="Capacity (Ah)", title="Capacity fade")
for a in ax:
    a.grid(alpha=.3)
ax[0].legend(fontsize=7)
ax[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig(CELL + "_plots.png", dpi=120)
print("saved", CELL + "_plots.png")