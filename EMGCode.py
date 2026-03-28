import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from hmmlearn import hmm

np.random.seed(42)

# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────
N = 300
t = np.linspace(0, 1, N)
phase = np.sin(np.pi * t)

elbow_angle  = 20 + 100 * phase
velocity     = np.gradient(elbow_angle)
biceps       = phase + 0.1 * np.random.randn(N)
triceps      = (1 - phase) + 0.1 * np.random.randn(N)
shoulder_imu = 0.2 * phase + 0.02 * np.random.randn(N)
forearm_imu  = 1.5 * phase + 0.1 * np.random.randn(N)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# 5 features — 3 directional, 2 energy
#
# Directional (signed, tell flex from ext):
#   signed_vel    : +ve during flexion, -ve during extension
#   muscle_signed : biceps - triceps, +ve during flexion
#   forearm_signed: forearm IMU signed mean, +ve during flexion
#
# Energy (magnitude, separate motion from rest):
#   vel_mag       : |velocity| — near zero at rest
#   emg_activity  : |biceps - triceps| — near zero at rest
# ─────────────────────────────────────────────
def extract_features(b, tr, f, v, window=12):
    feats = []
    for i in range(len(b) - window):
        bw = b[i:i+window]
        tw = tr[i:i+window]
        fw = f[i:i+window]
        vw = v[i:i+window]

        signed_vel     =  np.mean(vw)               # direction of motion
        muscle_signed  =  np.mean(bw) - np.mean(tw) # +ve = biceps dominant = flexion
        forearm_signed =  np.mean(fw)               # signed forearm displacement
        vel_mag        =  np.mean(np.abs(vw))       # energy — separates rest from motion
        emg_activity   =  np.abs(np.mean(bw) - np.mean(tw))  # muscle engagement level

        feats.append([signed_vel, muscle_signed, forearm_signed, vel_mag, emg_activity])
    return np.array(feats)

features = extract_features(biceps, triceps, forearm_imu, velocity)

# ─────────────────────────────────────────────
# NORMALISE features so HMM sees balanced scale
# ─────────────────────────────────────────────
feat_mean = features.mean(axis=0)
feat_std  = features.std(axis=0) + 1e-8
features_norm = (features - feat_mean) / feat_std

# ─────────────────────────────────────────────
# HMM — 3 STATES
# Initialise means to push states apart from the start
# so EM doesn't collapse flexion and extension together
# ─────────────────────────────────────────────
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=500,
    init_params="stmc",   # let us set means manually
    params="stmc"
)

# Warm-start means:
#   state 0 = REST     → all features near 0
#   state 1 = FLEXION  → signed features strongly positive
#   state 2 = EXTENSION→ signed features strongly negative
model.startprob_ = np.array([0.34, 0.33, 0.33])
model.transmat_  = np.array([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90],
])
model.means_ = np.array([
    [ 0.0,  0.0,  0.0,  0.0,  0.0],   # REST
    [ 1.5,  1.5,  1.5,  1.5,  1.5],   # FLEXION  — positive direction
    [-1.5, -1.5, -1.5,  1.5,  1.5],   # EXTENSION— negative direction, still active
])
model.covars_ = np.tile(np.eye(5) * 0.5, (3, 1, 1))

model.fit(features_norm)
probs = model.predict_proba(features_norm)   # (288, 3)

# ─────────────────────────────────────────────
# IDENTIFY STATES POST-TRAINING
# signed_vel is feature 0 — most reliable direction indicator
# vel_mag    is feature 3 — separates rest from motion
# ─────────────────────────────────────────────
signed_vel_means = model.means_[:, 0]
vel_mag_means    = model.means_[:, 3]

rest_state      = int(np.argmin(vel_mag_means))
remaining       = [i for i in range(3) if i != rest_state]
flexion_state   = remaining[int(np.argmax(signed_vel_means[remaining]))]
extension_state = remaining[int(np.argmin(signed_vel_means[remaining]))]

flex_prob = np.pad(probs[:, flexion_state],   (0, N - len(probs)), 'edge')
ext_prob  = np.pad(probs[:, extension_state], (0, N - len(probs)), 'edge')

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
BG     = "#f0f0f0"
AX_BG  = "#ffffff"
BORDER = "#a0a0a0"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    AX_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   "#222222",
    "axes.titlesize":    9,
    "axes.titleweight":  "normal",
    "axes.titlecolor":   "#222222",
    "axes.titlepad":     5,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "grid.color":        "#d0d0d0",
    "grid.linestyle":    "-",
    "text.color":        "#222222",
    "font.family":       "sans-serif",
    "font.size":         9,
    "legend.fontsize":   8,
    "legend.framealpha": 0.8,
    "legend.edgecolor":  BORDER,
})

# ─────────────────────────────────────────────
# WINDOW
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("EMG + IMU Intent Simulator")
root.configure(bg=BG)
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
root.bind("<F11>",    lambda e: root.attributes("-fullscreen", True))

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
left_frame  = tk.Frame(root, bg=BG, width=210)
left_frame.pack(side="left", fill="y", padx=(12, 6), pady=12)
left_frame.pack_propagate(False)

right_frame = tk.Frame(root, bg=BG)
right_frame.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)

# ─────────────────────────────────────────────
# LEFT PANEL
# ─────────────────────────────────────────────
tk.Label(left_frame, text="Biomechanics Monitor",
         bg=BG, fg="#222222", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 2))
tk.Label(left_frame, text="EMG · IMU · HMM Intent",
         bg=BG, fg="#888888", font=("Helvetica", 8)).pack(anchor="w", pady=(0, 12))

arm_fig, arm_ax = plt.subplots(figsize=(2.4, 2.1))
arm_fig.patch.set_facecolor(BG)
arm_canvas = FigureCanvasTkAgg(arm_fig, master=left_frame)
arm_canvas.get_tk_widget().pack(fill="x")

def draw_arm(angle):
    arm_ax.clear()
    arm_ax.set_facecolor(AX_BG)
    for sp in arm_ax.spines.values():
        sp.set_edgecolor(BORDER)
    shoulder = np.array([0.0, 0.0])
    elbow    = np.array([1.0, 0.0])
    rad      = np.radians(angle)
    wrist    = elbow + np.array([np.cos(rad), np.sin(rad)])
    arm_ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]],
                color="#333333", lw=5, solid_capstyle="round")
    arm_ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]],
                color="#0072bd", lw=5, solid_capstyle="round")
    for pt, c in [(shoulder, "#555"), (elbow, "#333"), (wrist, "#0072bd")]:
        arm_ax.scatter(*pt, s=50, color=c, zorder=5)
    arm_ax.set_xlim(-0.3, 2.3)
    arm_ax.set_ylim(-1.4, 1.4)
    arm_ax.set_title(f"Elbow angle: {int(angle)}°", fontsize=9)
    arm_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    arm_canvas.draw()

ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)

tk.Label(left_frame, text="IoM — Flexion",
         bg=BG, fg="#0072bd", font=("Helvetica", 8, "bold")).pack(anchor="w")
flex_label = tk.Label(left_frame, text="0.00",
                       bg=BG, fg="#0072bd", font=("Helvetica", 18, "bold"))
flex_label.pack(anchor="w")

tk.Label(left_frame, text="IoM — Extension",
         bg=BG, fg="#d95319", font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(6, 0))
ext_label = tk.Label(left_frame, text="0.00",
                      bg=BG, fg="#d95319", font=("Helvetica", 18, "bold"))
ext_label.pack(anchor="w")

state_label = tk.Label(left_frame, text="REST",
                        bg=BG, fg="#888888", font=("Helvetica", 9, "bold"))
state_label.pack(anchor="w", pady=(4, 12))

ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=6)

tk.Label(left_frame, text="Frame", bg=BG, fg="#666666",
         font=("Helvetica", 8)).pack(anchor="w", pady=(6, 0))
slider = ttk.Scale(left_frame, from_=0, to=N-1, orient="horizontal")
slider.pack(fill="x", pady=(2, 10))

auto = tk.BooleanVar(value=True)
ttk.Checkbutton(left_frame, text="Auto play", variable=auto).pack(anchor="w")

tk.Label(left_frame, text="\nEsc  exit fullscreen\nF11  fullscreen",
         bg=BG, fg="#bbbbbb", font=("Helvetica", 7)).pack(side="bottom", anchor="w")

# ─────────────────────────────────────────────
# RIGHT PANEL
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(11, 7))
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        hspace=0.75, wspace=0.32,
                        left=0.07, right=0.97,
                        top=0.95, bottom=0.06)

ax_emg  = fig.add_subplot(gs[0, 0])
ax_imu  = fig.add_subplot(gs[0, 1])
ax_enc  = fig.add_subplot(gs[1, 0])
ax_vel  = fig.add_subplot(gs[1, 1])
ax_flex = fig.add_subplot(gs[2, :])
ax_ext  = fig.add_subplot(gs[3, :])

canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

(ln_bic,)  = ax_emg.plot([], [], color="#0072bd", lw=1.2, label="Biceps")
(ln_tri,)  = ax_emg.plot([], [], color="#d95319", lw=1.2, label="Triceps")
(ln_sho,)  = ax_imu.plot([], [], color="#77ac30", lw=1.2, label="Shoulder")
(ln_fore,) = ax_imu.plot([], [], color="#edb120", lw=1.2, label="Forearm")
(ln_ang,)  = ax_enc.plot([], [], color="#0072bd", lw=1.2, label="Angle (°)")
(ln_vel,)  = ax_vel.plot([], [], color="#d95319", lw=1.2, label="Velocity")
(ln_flex,) = ax_flex.plot([], [], color="#0072bd", lw=1.5, label="IoM — Flexion")
(ln_ext,)  = ax_ext.plot( [], [], color="#d95319", lw=1.5, label="IoM — Extension")

vlines = [ax.axvline(0, color="#cccccc", lw=0.8, linestyle="--")
          for ax in [ax_emg, ax_imu, ax_enc, ax_vel, ax_flex, ax_ext]]

flex_fill = ax_flex.fill_between([], [], alpha=0.12, color="#0072bd")
ext_fill  = ax_ext.fill_between([],  [], alpha=0.12, color="#d95319")

def setup_ax(ax, title, ylabel=""):
    ax.set_title(title)
    ax.set_xlim(0, N)
    ax.set_xlabel("Sample", fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.grid(True, lw=0.5)
    ax.legend(loc="upper right")

setup_ax(ax_emg, "EMG Signals",           ylabel="Amplitude")
setup_ax(ax_imu, "IMU Signals",           ylabel="Amplitude")
setup_ax(ax_enc, "Joint Angle (Encoder)", ylabel="Degrees (°)")
setup_ax(ax_vel, "Angular Velocity",      ylabel="°/sample")

for ax, label, color in [
    (ax_flex, "IoM — Flexion",   "#0072bd"),
    (ax_ext,  "IoM — Extension", "#d95319"),
]:
    ax.set_title(label, color=color)
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sample", fontsize=7)
    ax.set_ylabel("P(state)", fontsize=7)
    ax.grid(True, lw=0.5)
    ax.axhline(0.5, color="#aaaaaa", lw=0.8, linestyle=":")
    ax.legend(loc="upper right")

for ax, data in [
    (ax_emg, np.concatenate([biceps, triceps])),
    (ax_imu, np.concatenate([shoulder_imu, forearm_imu])),
    (ax_enc, elbow_angle),
    (ax_vel, velocity),
]:
    margin = (data.max() - data.min()) * 0.12 or 0.1
    ax.set_ylim(data.min() - margin, data.max() + margin)

x_all = np.arange(N)

# ─────────────────────────────────────────────
# UPDATE LOOP
# ─────────────────────────────────────────────
idx = 0

def update():
    global idx, flex_fill, ext_fill

    if auto.get():
        idx = (idx + 1) % N
        slider.set(idx)
    else:
        idx = int(slider.get())

    x = x_all[:idx]

    ln_bic.set_data(x,  biceps[:idx])
    ln_tri.set_data(x,  triceps[:idx])
    ln_sho.set_data(x,  shoulder_imu[:idx])
    ln_fore.set_data(x, forearm_imu[:idx])
    ln_ang.set_data(x,  elbow_angle[:idx])
    ln_vel.set_data(x,  velocity[:idx])
    ln_flex.set_data(x, flex_prob[:idx])
    ln_ext.set_data(x,  ext_prob[:idx])

    flex_fill.remove()
    ext_fill.remove()
    flex_fill = ax_flex.fill_between(x, flex_prob[:idx], alpha=0.12, color="#0072bd")
    ext_fill  = ax_ext.fill_between(x,  ext_prob[:idx],  alpha=0.12, color="#d95319")

    for vl in vlines:
        vl.set_xdata([idx, idx])

    canvas.draw_idle()

    if idx % 3 == 0:
        draw_arm(elbow_angle[idx])

    fp = flex_prob[idx]
    ep = ext_prob[idx]
    flex_label.config(text=f"{fp:.2f}")
    ext_label.config(text=f"{ep:.2f}")

    if fp > 0.5:
        state_label.config(text="FLEXION",   fg="#0072bd")
    elif ep > 0.5:
        state_label.config(text="EXTENSION", fg="#d95319")
    else:
        state_label.config(text="REST",      fg="#888888")

    root.after(40, update)

update()
root.mainloop()