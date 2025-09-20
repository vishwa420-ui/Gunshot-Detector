import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.optimize import least_squares

# ----------------------------
# Configuration
# ----------------------------
NUM_MICS = 4
MIC_RADIUS = 0.2             # meters
SOUND_SPEED = 343.0          # m/s
FS = 48000                   # Hz
NOISE_LEVEL = 0.002
PULSE_DURATION = 0.003
PULSE_CENTER = 0.001
PULSE_SIGMA = 0.00025

angles = np.linspace(0, 2*np.pi, NUM_MICS, endpoint=False)
mic_positions = np.stack([MIC_RADIUS * np.cos(angles),
                          MIC_RADIUS * np.sin(angles)], axis=1)

def simulate_and_estimate(src_pos):
    """Simulate signals for a given source position and estimate back."""
    pulse_len = int(PULSE_DURATION * FS)
    t_pulse = np.arange(pulse_len) / FS
    pulse = np.exp(-((t_pulse - PULSE_CENTER) / PULSE_SIGMA)**2)
    pulse = pulse / np.max(pulse)

    distances = np.linalg.norm(mic_positions - src_pos, axis=1)
    delays = distances / SOUND_SPEED
    delay_samples = np.round(delays * FS).astype(int)

    N = pulse_len + delay_samples.max() + int(0.01 * FS)
    mic_signals = np.zeros((NUM_MICS, N))
    for i, ds in enumerate(delay_samples):
        start = ds
        end = start + pulse_len
        mic_signals[i, start:end] += pulse
        mic_signals[i] *= 1.0 + 0.05*(np.random.rand()-0.5)
        mic_signals[i] += np.random.normal(0, NOISE_LEVEL, size=N)

    envelopes = np.abs(hilbert(mic_signals, axis=1))
    arrival_indices = np.argmax(envelopes, axis=1)
    arrival_times = arrival_indices / FS
    ref_idx = np.argmin(arrival_indices)
    tdoas = arrival_times - arrival_times[ref_idx]

    def residuals(p):
        x, y = p
        dists = np.hypot(x - mic_positions[:,0], y - mic_positions[:,1])
        pred = (dists - dists[ref_idx]) / SOUND_SPEED
        return pred - tdoas

    first_angle = angles[ref_idx]
    approx_dist = arrival_times[ref_idx] * SOUND_SPEED
    p0 = np.array([approx_dist * np.cos(first_angle),
                   approx_dist * np.sin(first_angle)])

    res = least_squares(residuals, p0, method='lm')
    est_x, est_y = res.x
    return est_x, est_y

# ----------------------------
# Interactive Plot Setup
# ----------------------------
fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(7,7))
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# Add compass labels
r_max = 10  # adjust based on max expected distance
ax.text(0, r_max, "N", ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.text(np.deg2rad(90), r_max, "E", ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.text(np.deg2rad(180), r_max, "S", ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.text(np.deg2rad(270), r_max, "W", ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.scatter(angles, [MIC_RADIUS]*NUM_MICS, label='microphones', s=40)
ax.set_title("Click anywhere to simulate a gunshot source")

true_point = ax.scatter([], [], marker='x', c='green', s=80, label='true source')
est_point = ax.scatter([], [], marker='o', c='red', s=80, label='estimated source')
distance_text = ax.text(0, 0, "", color="red", fontsize=12, weight="bold",
                        ha="center", va="bottom")
ax.legend(loc='lower left')

def onclick(event):
    if event.inaxes != ax:
        return
    theta = event.xdata
    r = event.ydata
    if r is None or theta is None:
        return

    src_pos = np.array([r * np.cos(theta), r * np.sin(theta)])
    est_x, est_y = simulate_and_estimate(src_pos)
    est_r = np.hypot(est_x, est_y)
    est_theta = np.arctan2(est_y, est_x)

    print(f"Clicked: {r:.2f} m @ {np.degrees(theta):.1f}° | "
          f"Estimated: {est_r:.2f} m @ {(np.degrees(est_theta)%360):.1f}°")

    true_point.set_offsets(np.c_[theta, r])
    est_point.set_offsets(np.c_[est_theta, est_r])

    distance_text.set_position((est_theta, est_r))
    distance_text.set_text(f"{est_r:.2f} m")

    ax.set_title(f"Clicked: {r:.2f} m | Estimated: {est_r:.2f} m @ {(np.degrees(est_theta)%360):.1f}°")
    fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
