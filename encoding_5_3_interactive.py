"""Section 5.3 - Analog Data, Digital Signals

Analog-to-digital encoding simulator:
  * PCM  (sampling -> quantization -> binary encoding -> reconstruction)
  * Delta modulation (staircase, slope overload vs. granular noise)
  * PCM vs DM comparison, plus optional companding / nonlinear encoding
"""
import math
import matplotlib.pyplot as plt

MU = 255


# ---------------------------------------------------------------- input
def ask(prompt, default, cast=float):
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if text == "":
            return default
        try:
            return cast(text)
        except ValueError:
            print("  invalid, try again")


def make_curve(kind, A, fm):
    if kind == 2:                                       # square
        return lambda t: A if math.sin(2 * math.pi * fm * t) >= 0 else -A
    if kind == 3:                                       # triangle
        return lambda t: 2 * A / math.pi * math.asin(math.sin(2 * math.pi * fm * t))
    return lambda t: A * math.sin(2 * math.pi * fm * t)  # sine


def get_signal():
    """Returns (samples, fs, curve). curve is None for user-typed values."""
    kind = ask("Signal: 1 = sine, 2 = square, 3 = triangle, 4 = my own values", 1, int)
    while kind not in (1, 2, 3, 4):
        print("  choose 1, 2, 3, or 4")
        kind = ask("Signal", 1, int)
    if kind == 4:
        vals = []
        while len(vals) < 2:
            vals = [float(v) for v in input("Values (at least 2): ").replace(",", " ").split()]
        fs = ask("fs (Hz)", 8000.0)
        while fs <= 0:
            print("  fs must be greater than zero")
            fs = ask("fs (Hz)", 8000.0)
        print(f"\nTs = {1 / fs:g} s")
        return vals, fs, None
    A = ask("Amplitude (V)", 4.0)
    while A <= 0:
        print("  amplitude must be greater than zero")
        A = ask("Amplitude (V)", 4.0)
    fm = ask("Frequency (Hz)", 1000.0)
    while fm <= 0:
        print("  frequency must be greater than zero")
        fm = ask("Frequency (Hz)", 1000.0)
    fs = ask("fs (Hz)", 4000.0)
    while fs <= 0:
        print("  fs must be greater than zero")
        fs = ask("fs (Hz)", 4000.0)
    n = max(2, ask("Samples", 16, int))
    curve = make_curve(kind, A, fm)
    print(f"\nTs = 1/fs = {1 / fs:g} s | Nyquist rate 2fm = {2 * fm:g} Hz | "
          + ("OK (fs > 2fm)" if fs > 2 * fm else "ALIASING (fs <= 2fm)"))
    if kind in (2, 3):
        print("  (square/triangle contain harmonics above fm, so the real Nyquist limit is stricter)")
    return [curve(k / fs) for k in range(n)], fs, curve


# ---------------------------------------------------------------- helpers
def snr_db(x, y):
    noise = sum((a - b) ** 2 for a, b in zip(x, y))
    if noise == 0:
        return math.inf
    power = sum(a * a for a in x)
    return -math.inf if power == 0 else 10 * math.log10(power / noise)


def compress(u):
    s = 2 * u - 1
    return (math.copysign(math.log1p(MU * abs(s)) / math.log1p(MU), s) + 1) / 2


def expand(v):
    s = 2 * v - 1
    return (math.copysign(((1 + MU) ** abs(s) - 1) / MU, s) + 1) / 2


def plot_bits(ax, bits, title):
    ax.step(range(len(bits) + 1), bits + bits[-1:], where="post")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.3, 1.3)
    ax.set_title(title)
    ax.set_xlabel("bit")


def plot_analog(ax, vals, fs, curve, samples=True, faint=False):
    """Plot the original analog curve consistently, plus optional PAM samples."""
    t = [k / fs for k in range(len(vals))]
    if curve:
        dense = [i * t[-1] / 500 for i in range(501)]
        ax.plot(dense, [curve(x) for x in dense], color="dimgray", ls="--",
                lw=1.2, alpha=0.85 if faint else 1.0, label="analog")
    if samples:
        ax.plot(t, vals, "o", color="darkorange", markersize=5,
                label="samples (PAM)")
    ax.set_xlabel("time (s)")


def new_window(title, rows):
    """One window holding one separate graph per waveform."""
    fig, axes = plt.subplots(rows, 1, figsize=(10, min(11, 2.8 * rows)), squeeze=False)
    fig.suptitle(title, fontweight="bold")
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    return fig, list(axes[:, 0])


def finish(fig):
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# ---------------------------------------------------------------- PCM
def pcm_levels(n, lo, hi, companded=False):
    """Reconstruction voltage of every code 0..2^n - 1."""
    L = 2 ** n
    return [lo + (expand((c + 0.5) / L) if companded else (c + 0.5) / L) * (hi - lo)
            for c in range(L)]


def pcm(vals, n, lo, hi, companded=False):
    """Returns (codes, reconstructed values). Mid-rise quantizer with L = 2^n levels."""
    L = 2 ** n
    levels = pcm_levels(n, lo, hi, companded)
    codes = []
    for x in vals:
        u = min(1.0, max(0.0, (x - lo) / (hi - lo)))   # normalize to 0..1
        if companded:
            u = compress(u)
        codes.append(min(L - 1, int(u * L)))           # quantize
    return codes, [levels[c] for c in codes]


def run_pcm(vals, fs, curve, companded=False):
    n = ask("Bits per sample (n)", 4, int)
    while not 1 <= n <= 16:
        print("  choose between 1 and 16 bits")
        n = ask("Bits per sample (n)", 4, int)
    lo = ask("x_min (V)", min(vals))
    hi = ask("x_max (V)", max(vals))
    if hi <= lo:
        print("  x_max must be greater than x_min")
        return
    L = 2 ** n
    codes, rec = pcm(vals, n, lo, hi, companded)
    words = [format(c, f"0{n}b") for c in codes]

    print(f"\nTs = {1 / fs:g} s | L = 2^{n} = {L} levels"
          + ("" if companded else f" | step = {(hi - lo) / L:.4f} V"))
    print(f"\n{'k':>3} {'PAM':>8} {'code':>5} {'PCM':>{n}} {'recon':>8} {'error':>8}")
    for k, (x, c, w, r) in enumerate(zip(vals, codes, words, rec)):
        print(f"{k:>3} {x:>8.3f} {c:>5} {w:>{n}} {r:>8.3f} {r - x:>+8.3f}")

    snr = snr_db(vals, rec)
    print(f"\nBits: {''.join(words)}")
    print(f"Rb  = fs x n = {fs:g} x {n} = {n * fs:g} bps")
    print(f"SNR = {snr:.2f} dB", end="")
    if companded:
        uni = snr_db(vals, pcm(vals, n, lo, hi)[1])
        print(f"  (uniform PCM: {uni:.2f} dB -> companding gain {snr - uni:+.2f} dB)")
    else:
        print(f"  (theory 6.02n + 1.76 = {6.02 * n + 1.76:.2f} dB)")

    t = [k / fs for k in range(len(vals))]
    kind = "Companded PCM" if companded else "PCM"

    fig, (a1, a2, a3) = new_window(kind, 3)
    a1.set_title("Analog signal and PAM samples")
    plot_analog(a1, vals, fs, curve)
    a1.legend()

    a2.set_title(f"Quantized waveform ({L} levels)")
    plot_analog(a2, vals, fs, curve, samples=False, faint=True)
    if L <= 32:
        for lv in pcm_levels(n, lo, hi, companded):
            a2.axhline(lv, color="gray", lw=0.4, alpha=0.6)
    a2.step(t, rec, where="post", label="PCM quantized")
    a2.legend()

    plot_bits(a3, [int(b) for b in "".join(words)], "Digital output (NRZ-L)")
    finish(fig)


# ---------------------------------------------------------------- Delta modulation
def dm(vals, delta, s0):
    """Returns (bits, staircase). stairs[k] is the staircase after sample k."""
    s, bits, stairs = s0, [], [s0]
    for x in vals[1:]:
        bit = 1 if x >= s else 0
        s += delta if bit else -delta
        bits.append(bit)
        stairs.append(s)
    return bits, stairs


def overload_idx(vals, stairs, delta):
    """Samples where the staircase was more than one step away (slope overload);
    every other sample only shows granular (quantizing) noise."""
    return [k for k in range(1, len(vals)) if abs(vals[k] - stairs[k - 1]) > delta]


def run_dm(vals, fs, curve):
    delta = ask("Step size delta (V)", 1.0)
    while delta <= 0:
        print("  delta must be greater than zero")
        delta = ask("Step size delta (V)", 1.0)
    s0 = ask("Initial staircase s(0) (V)", vals[0])
    bits, stairs = dm(vals, delta, s0)
    ov = overload_idx(vals, stairs, delta)
    err = [s - x for x, s in zip(vals, stairs)]
    snr = snr_db(vals, stairs)

    print(f"\n{'k':>3} {'m(kTs)':>8} {'s(k)':>8} {'bit':>4} {'error':>8}")
    for k in range(1, len(vals)):
        print(f"{k:>3} {vals[k]:>8.2f} {stairs[k]:>8.2f} {bits[k - 1]:>4} {err[k]:>+8.2f}"
              + ("  <- overload" if k in ov else ""))
    print(f"\nBits: {''.join(map(str, bits))}")
    print(f"Rb  = fs x 1 bit = {fs:g} bps")
    print(f"SNR = {snr:.2f} dB")

    worst = max(abs(b - a) for a, b in zip(vals, vals[1:]))
    print(f"Max signal change {worst:.3f} V/sample vs step {delta:g} V/sample "
          f"({worst * fs:.1f} vs {delta * fs:.1f} V/s) -> "
          + ("SLOPE OVERLOAD" if worst > delta else "OK, staircase can follow"))
    print(f"Samples flagged for possible slope overload: {len(ov)}")
    print("Note: remaining samples are not automatically classified as granular-noise samples.")

    t = [k / fs for k in range(len(vals))]
    fig, (a1, a2) = new_window("Delta modulation", 2)
    a1.set_title("Analog signal and staircase")
    plot_analog(a1, vals, fs, curve)
    a1.step(t, stairs, where="post", label="staircase")
    if ov:
        a1.plot([t[k] for k in ov], [vals[k] for k in ov], "rx", ms=9, label="slope overload")
    a1.legend()

    plot_bits(a2, bits, "Digital output (1 = up, 0 = down)")
    finish(fig)


def run_dm_steps(vals, fs, curve):
    """Shows how the step size changes the staircase and its errors."""
    delta = ask("Base step size delta (V)", 1.0)
    while delta <= 0:
        print("  delta must be greater than zero")
        delta = ask("Base step size delta (V)", 1.0)
    s0 = ask("Initial staircase s(0) (V)", vals[0])
    t = [k / fs for k in range(len(vals))]
    fig, axes = new_window("DM step-size effect", 3)
    print(f"\n{'delta (V)':>10}{'SNR (dB)':>10}{'overload':>10}{'granular':>10}")
    for a, d in zip(axes, (delta / 2, delta, delta * 2)):
        _, stairs = dm(vals, d, s0)
        ov = overload_idx(vals, stairs, d)
        gran = len(vals) - 1 - len(ov)
        s = snr_db(vals, stairs)
        print(f"{d:>10g}{s:>10.2f}{len(ov):>10}{gran:>10}")
        a.set_title(f"delta = {d:g} V | SNR {s:.2f} dB | slope overload {len(ov)} | granular {gran}")
        plot_analog(a, vals, fs, curve)
        a.step(t, stairs, where="post", label="staircase")
        a.legend()
    finish(fig)


# ---------------------------------------------------------------- PCM vs DM
def make_fine_signal(vals, fs, curve, factor):
    """Create the reference signal at the faster DM sampling rate."""
    fine_count = (len(vals) - 1) * factor + 1
    fine_times = [j / (fs * factor) for j in range(fine_count)]
    if curve is not None:
        fine = [curve(t) for t in fine_times]
    else:
        fine = []
        for p, q in zip(vals, vals[1:]):
            fine.extend(p + (q - p) * j / factor for j in range(factor))
        fine.append(vals[-1])
    return fine_times, fine


def best_dm(vals, fs, curve, factor, span):
    """Find a useful DM step size and return its full-rate staircase."""
    fine_times, fine = make_fine_signal(vals, fs, curve, factor)
    best = (-math.inf, None, None)
    # Search a broad, logarithmic range of fixed DM step sizes.
    for i in range(60):
        delta = span * 0.0005 * (1.25 ** i)
        _, stairs = dm(fine, delta, fine[0])
        reconstructed_at_original_rate = stairs[::factor]
        score = snr_db(vals, reconstructed_at_original_rate)
        if score > best[0]:
            best = (score, stairs, delta)
    score, stairs, delta = best
    return score, fine_times, fine, stairs, delta


def run_compare(vals, fs, curve):
    """Visual-only comparison at matched PCM/DM bit rates."""
    n = ask("PCM bits per sample (DM uses the same factor)", 4, int)
    if not 1 <= n <= 10:
        print("  choose between 1 and 10 bits for this comparison")
        return
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        print("  signal must have a nonzero amplitude range")
        return

    # PCM uses n bits at fs. DM uses one bit at n*fs, so both have n*fs bps.
    pcm_rec = pcm(vals, n, lo, hi)[1]
    auto_snr, dm_times, fine, auto_stairs, auto_delta = best_dm(
        vals, fs, curve, n, hi - lo
    )

    # Optional user override: Enter keeps the automatically selected step size.
    while True:
        raw_delta = input(
            f"DM step size delta in V [Enter = automatic {auto_delta:.4g}]: "
        ).strip()
        if raw_delta == "":
            dm_delta = auto_delta
            dm_stairs = auto_stairs
            break
        try:
            dm_delta = float(raw_delta)
            if not math.isfinite(dm_delta) or dm_delta <= 0:
                raise ValueError
            _, dm_stairs = dm(fine, dm_delta, fine[0])
            break
        except ValueError:
            print("  enter a positive number, or press Enter for automatic")

    pcm_rate = n * fs
    dm_snr = snr_db(vals, dm_stairs[::n])
    dm_fs = n * fs

    print("\nVisual comparison only: same analog input and time interval.")
    print(f"PCM: {n} bits/sample x {fs:g} samples/s = {pcm_rate:g} bps")
    print(f"DM:  1 bit/sample x {dm_fs:g} samples/s = {pcm_rate:g} bps")
    print(f"DM step size: delta = {dm_delta:.4g} V"
          + (" (automatic selection)" if raw_delta == "" else " (user-selected)"))
    print(f"DM reconstruction SNR at the original sample instants: {dm_snr:.2f} dB")

    # Three aligned panels. Use consistent colors and identical time/amplitude scales.
    t = [k / fs for k in range(len(vals))]
    fig, axes = new_window("Visual comparison: PCM vs. DM", 3)

    axes[0].set_title(f"Original analog input (PAM sampling rate = {fs:g} Hz)")
    plot_analog(axes[0], vals, fs, curve, samples=True)
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].legend(fontsize=8)

    axes[1].set_title(f"PCM reconstruction ({n}-bit amplitude quantization)")
    plot_analog(axes[1], vals, fs, curve, samples=False, faint=True)
    axes[1].step(t, pcm_rec, where="post", color="tab:blue",
                 linewidth=1.5, label=f"PCM ({n} bits/sample)")
    axes[1].set_ylabel("Amplitude (V)")
    axes[1].legend(fontsize=8)

    axes[2].set_title(
        f"DM staircase (sampling rate = {dm_fs:g} Hz; delta = {dm_delta:.4g} V; "
        f"matched {pcm_rate:g} bps)"
    )
    plot_analog(axes[2], vals, fs, curve, samples=False, faint=True)
    axes[2].step(dm_times, dm_stairs, where="post", color="tab:green",
                 linewidth=1.5, label="DM staircase")
    axes[2].set_ylabel("Amplitude (V)")
    axes[2].legend(fontsize=8)

    # Include any DM overshoot while keeping one common vertical scale.
    y_min = min(lo, min(fine), min(pcm_rec), min(dm_stairs))
    y_max = max(hi, max(fine), max(pcm_rec), max(dm_stairs))
    margin = max((y_max - y_min) * 0.06, 0.1)
    for ax in axes:
        ax.set_xlim(t[0], t[-1] if len(t) > 1 else t[0] + 1 / fs)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, alpha=0.2)
    finish(fig)


# ---------------------------------------------------------------- SNR vs bits
def run_snr_table(vals, fs):
    lo, hi = min(vals), max(vals)
    if hi == lo:
        print("  signal is constant, nothing to measure")
        return
    ns = list(range(1, 13))
    theory = [6.02 * n + 1.76 for n in ns]
    uni = [snr_db(vals, pcm(vals, n, lo, hi)[1]) for n in ns]
    com = [snr_db(vals, pcm(vals, n, lo, hi, True)[1]) for n in ns]
    print(f"\n{'n':>3}{'L':>6}{'Rb (bps)':>10}{'theory dB':>11}{'uniform':>10}{'companded':>11}")
    for i, n in enumerate(ns):
        print(f"{n:>3}{2 ** n:>6}{n * fs:>10g}{theory[i]:>11.2f}{uni[i]:>10.2f}{com[i]:>11.2f}")
    print("Each extra bit adds about 6 dB. Measured values use only your samples, so they")
    print("differ from theory (which assumes a full-scale sine and many samples).")



# ---------------------------------------------------------------- companding curve
def show_curve():
    print(f"\n{'input':>7}{'no companding':>15}{'mu = 10':>10}{'mu = 255':>10}")
    for i in range(11):
        x = i / 10
        c10, c255 = (math.log1p(mu * x) / math.log1p(mu) for mu in (10, MU))
        print(f"{x:>7.1f}{x:>15.3f}{c10:>10.3f}{c255:>10.3f}")
    print("Small inputs are boosted more than large ones, so quiet sounds get finer steps.")


# ---------------------------------------------------------------- main
MENU = [
    ("1. PCM", "View PCM process individually"),
    ("2. Delta Modulation", "View DM process individually"),
    ("3. Compare PCM vs. DM", "Display both outputs simultaneously"),
    ("4. PCM with Companding", "Explore nonlinear encoding"),
    ("5. Companding Curve", "View companding behavior"),
    ("6. SNR vs. Bits", "Analyze PCM performance"),
    ("7. DM Step-Size Effect", "Analyze DM performance"),
    ("8. New Signal", "Enter a different input signal"),
    ("0. Quit", "Exit the program"),
]


def print_menu():
    print(f"\n{'Menu':<26}Function")
    print("-" * 62)
    for name, function in MENU:
        print(f"{name:<26}{function}")


def main():
    signal = get_signal()
    while True:
        print_menu()
        vals, fs, curve = signal
        c = ask("Choose", 1, int)
        if c == 1:
            run_pcm(vals, fs, curve)
        elif c == 2:
            run_dm(vals, fs, curve)
        elif c == 3:
            run_compare(vals, fs, curve)
        elif c == 4:
            run_pcm(vals, fs, curve, companded=True)
        elif c == 5:
            show_curve()
        elif c == 6:
            run_snr_table(vals, fs)
        elif c == 7:
            run_dm_steps(vals, fs, curve)
        elif c == 8:
            signal = get_signal()
        elif c == 0:
            break
        else:
            print("  invalid choice, pick a number from the menu")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram exited.")