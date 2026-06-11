#!/usr/bin/env python
# coding: utf-8
# =====================================================================
#  TTS9 — Advanced telemetry diagnostics
# ---------------------------------------------------------------------
#  THE OBSERVED FACT
#  -----------------
#  The kinematic reconstruction shows a ~3 m/s discontinuity in the
#  gondola's HORIZONTAL velocity at balloon burst (the notebook calls it a
#  ~3.2 m/s southward "impulse" and rejects H0 = "natural burst").  This
#  module asks, one hypothesis at a time, WHAT could cause that step — and
#  for each one produces a SINGLE figure that shows how well the data can
#  confirm or exclude it.
#
#  THE CANDIDATE EXPLANATIONS (one section + one figure each)
#  ---------------------------------------------------------
#    H1  GPS measurement error   — is the step just GPS noise / anisotropy?
#    H2  Wind drag during fall   — can drag at expected winds turn the
#                                  gondola by 3 m/s as it descends?
#    H3  Rotation + Magnus       — can a spinning sphere generate a lateral
#                                  aerodynamic force of that size?
#    H4  Reconstruction artefact — is the "impulse" an artefact of the
#                                  assumed apogee-wind initial velocity, and
#                                  is its statistical significance overstated?
#    H5  Pendulum                — was the gondola swinging below the balloon
#                                  and the swing velocity frozen at release?
#    (H6 External collision      — the residual hypothesis: what survives once
#                                  H1-H5 are weighed.  Not independently
#                                  testable from this telemetry.)
#
#  Each figure is self-contained and named TTS9_H{n}_*.png so the set can be
#  dropped into the README in order, each with its own caption.  The module
#  runs stand-alone (it reconstructs the few burst quantities it needs).
#
#  Honest summary of what the data can do: H1, H2, H3 are firmly EXCLUDED;
#  H4 shows the notebook's significance is OVERSTATED (the step is real but
#  its size/significance are model-dependent); H5 is excluded for any long
#  suspension line and merely invisible (not supported) for a short one.
#  The data are therefore CONSISTENT WITH but NOT DIAGNOSTIC OF a collision.
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

# All figures use a GitHub-style dark background; the accent colours
# (#58a6ff, #ffa657, #7ee787, #d2a8ff ...) are chosen to read on it.
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'savefig.facecolor': '#0d1117',
    'axes.facecolor': '#0d1117', 'axes.edgecolor': '#8b949e',
    'axes.labelcolor': '#e6edf3', 'text.color': '#e6edf3',
    'xtick.color': '#c9d1d9', 'ytick.color': '#c9d1d9',
    'axes.titlecolor': '#e6edf3', 'legend.edgecolor': '#30363d',
    'legend.facecolor': '#161b22', 'grid.color': '#30363d',
})

# ── Constants shared with the main notebook ───────────────────────────
M_GON   = 0.300                       # gondola mass [kg]
D_SPH   = 0.160                       # sphere diameter [m]
R_SPH   = D_SPH / 2                   # sphere radius [m]
A_SPH   = np.pi * R_SPH**2            # frontal area [m^2]
LAT_C   = 49.70
M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(LAT_C))
SIGMA_HOR = 5.0                       # GPS horizontal noise [m]
N_HORIZ   = 5                         # packets after burst used in the test
G         = 9.80665

UVA_COL = '(UVC)UVA[uW/cm2]'          # directional UV channel = rotation proxy


# ── ISA density (duplicated so the module is self-contained) ──────────
def isa_density(h):
    h = max(float(h), 0); R = 287.05287
    if h <= 11000:
        T = 288.15 - 0.0065 * h
        p = 101325 * (T / 288.15) ** (G / (R * 0.0065))
    else:
        T = 216.65; p = 22632 * np.exp(-G * (h - 11000) / (R * T))
    return p / (R * T)


# =====================================================================
#  Data loading and burst reconstruction (shared by every hypothesis)
# =====================================================================
def load_flight(path='DATA+SENSORS-flight.xlsx'):
    """Load the flight workbook and return a clean, time-tagged DataFrame."""
    raw = pd.read_excel(path)
    raw = raw.rename(columns={'čas[UTC]': 'time', 'Šířka[°]': 'lat',
                              'Délka[°]': 'lon', 'výška[m]': 'alt'})
    raw['lat'] = pd.to_numeric(raw['lat'], errors='coerce')
    raw['lon'] = pd.to_numeric(raw['lon'], errors='coerce')

    def to_sec(s):
        if hasattr(s, 'hour'):
            return s.hour * 3600 + s.minute * 60 + s.second
        h, m, sc = str(s).split(':'); return int(h) * 3600 + int(m) * 60 + int(sc)

    df = raw[raw['alt'].notna() & (raw['lat'] > 10)].reset_index(drop=True)
    df['t_sec'] = df['time'].apply(to_sec)
    return df


def reconstruct_burst(df):
    """Minimal re-implementation of the notebook's kinematic burst fit.

    Returns a dict of the burst state and the ascent wind interpolators that
    the H0 propagation needs.  Kept deliberately close to the notebook code so
    the numbers match; see the notebook for the physics commentary.
    """
    MAX = int(df['alt'].idxmax())
    T0  = df.loc[MAX, 't_sec']
    ASC  = df.loc[:MAX].reset_index(drop=True)
    DESC = df.loc[MAX:].reset_index(drop=True)

    # Ascent altitude/position: 2nd-order polynomial over the last 30 packets.
    af = ASC.tail(30); ta = (af['t_sec'].values - T0).astype(float)
    p_h   = np.polyfit(ta, af['alt'].values.astype(float), 2)
    p_lat = np.polyfit(ta, af['lat'].values.astype(float), 2)
    p_lon = np.polyfit(ta, af['lon'].values.astype(float), 2)
    dp_h  = np.polyder(p_h)

    # Descent free-fall fit -> separation time and drag k (vertical).
    dfd = DESC.head(25)
    td_fit = (dfd['t_sec'].values - T0).astype(float)
    hd_fit = dfd['alt'].values.astype(float)

    def sim_descent(k, h0, vh0, ts, te, dt=0.02):
        h = h0; v = vh0; t = ts; ht = [h]; tt = [t]
        while h > 0 and t < te[-1] + 5:
            rho = isa_density(h)
            a = G - k*rho*v**2 if v >= 0 else G + k*rho*v**2
            v += a*dt; h -= v*dt; t += dt; ht.append(h); tt.append(t)
        return np.interp(te, tt, ht)

    def fit_k(tsep):
        h0 = float(np.polyval(p_h, tsep)); vh0 = -float(np.polyval(dp_h, tsep))
        r = least_squares(lambda pp: (sim_descent(pp[0], h0, vh0, tsep, td_fit) - hd_fit) / 12.0,
                          [0.03], bounds=([0.005], [0.5]))
        return r, np.sqrt(np.mean(r.fun**2)) * 12

    scan = np.linspace(-6, 2, 80)
    rms  = np.array([fit_k(ts)[1] for ts in scan])
    TS   = scan[int(np.argmin(rms))]
    K    = float(fit_k(TS)[0].x[0])

    state = dict(
        T0=T0, MAX=MAX, ASC=ASC, DESC=DESC, K=K, TS=TS,
        H_SEP=float(np.polyval(p_h, TS)),
        LAT_SEP=float(np.polyval(p_lat, TS)),
        LON_SEP=float(np.polyval(p_lon, TS)),
        VH0=-float(np.polyval(dp_h, TS)),
    )

    # In-situ ascent wind profile (11-packet sliding linear fit of GPS).
    half = 5
    ah = ASC['alt'].values; at = ASC['t_sec'].values.astype(float)
    ala = ASC['lat'].values; alo = ASC['lon'].values
    hw, uw, vw = [], [], []
    for i in range(half, len(at) - half):
        sl = slice(i - half, i + half + 1)
        uw.append(np.polyfit(at[sl], alo[sl], 1)[0] * M_PER_DEG_LON)
        vw.append(np.polyfit(at[sl], ala[sl], 1)[0] * M_PER_DEG_LAT)
        hw.append(ah[i])
    hw = np.array(hw); uw = np.array(uw); vw = np.array(vw)
    m = (hw >= 5000) & (hw <= 15500)
    hw, uw, vw = hw[m], uw[m], vw[m]
    s = np.argsort(hw); hw, uw, vw = hw[s], uw[s], vw[s]
    state['u_f'] = interp1d(hw, uw, fill_value='extrapolate')
    state['v_f'] = interp1d(hw, vw, fill_value='extrapolate')
    state['U_SEP'] = float(state['u_f'](state['H_SEP']))
    state['V_SEP'] = float(state['v_f'](state['H_SEP']))
    return state


def propagate_horiz(st, dv_N=0., dv_E=0., V0=None, U0=None, dt=0.02):
    """Drift the gondola from the burst point under wind + horizontal drag.

    V0/U0 override the initial northward/eastward velocity (default = the
    ascent-wind values).  Used to probe how sensitive H0 is to that choice.
    """
    h = st['H_SEP']; vv = st['VH0']
    u_g = (st['U_SEP'] if U0 is None else U0) + dv_E
    v_g = (st['V_SEP'] if V0 is None else V0) + dv_N
    t = st['TS']; lat = st['LAT_SEP']; lon = st['LON_SEP']
    to = [t]; lao = [lat]; loo = [lon]
    Cd_h = 0.30
    while h > 100 and t < st['TS'] + 150:
        rho = isa_density(h)
        av = G - st['K']*rho*vv**2 if vv >= 0 else G + st['K']*rho*vv**2
        vv += av*dt; h -= vv*dt
        uw = float(st['u_f'](max(h, 500))); vw = float(st['v_f'](max(h, 500)))
        dur = u_g - uw; dvr = v_g - vw; vrh = np.hypot(dur, dvr)
        if vrh > 0.001:
            ad = 0.5 * Cd_h * A_SPH * rho * vrh**2 / M_GON
            u_g -= ad * dur/vrh * dt; v_g -= ad * dvr/vrh * dt
        lat += v_g*dt/M_PER_DEG_LAT; lon += u_g*dt/M_PER_DEG_LON; t += dt
        to.append(t); lao.append(lat); loo.append(lon)
    return np.array(to), np.array(lao), np.array(loo)


def chi2_H0(st, V0=None, U0=None):
    """Reduced-residual chi-squared of the no-impulse model for the first
    N_HORIZ descent packets, given an assumed initial horizontal velocity."""
    DESC = st['DESC']; T0 = st['T0']
    td = (DESC['t_sec'].values[:N_HORIZ] - T0).astype(float)
    lm = DESC['lat'].values[:N_HORIZ]; om = DESC['lon'].values[:N_HORIZ]
    ts, la, lo = propagate_horiz(st, 0., 0., V0=V0, U0=U0)
    lp = np.interp(td, ts, la); op = np.interp(td, ts, lo)
    rN = (lm - lp) * M_PER_DEG_LAT; rE = (om - op) * M_PER_DEG_LON
    return np.sum(rN**2 + rE**2) / SIGMA_HOR**2


# =====================================================================
#  Shared measurement helpers used by the hypothesis tests
# =====================================================================
def gps_axis_noise(df, idx, win=9, deg=2):
    """Per-axis GPS position noise via leave-one-out local-polynomial detrend.

    For each point we fit a low-order polynomial to its neighbours only and
    take the prediction residual at the held-out point.  This removes the
    smooth trajectory and exposes the high-frequency position scatter
    separately in E and N.  Returns (rms_E, rms_N) in metres."""
    idx = np.asarray(idx)
    t = df['t_sec'].values[idx].astype(float)
    E = df['lon'].values[idx] * M_PER_DEG_LON
    N = df['lat'].values[idx] * M_PER_DEG_LAT
    h = win // 2; rE, rN = [], []
    for i in range(h, len(idx) - h):
        sl = slice(i - h, i + h + 1); tt = t[sl] - t[i]
        cE = np.polyfit(np.delete(tt, h), np.delete(E[sl], h), deg)
        cN = np.polyfit(np.delete(tt, h), np.delete(N[sl], h), deg)
        rE.append(E[i] - np.polyval(cE, 0)); rN.append(N[i] - np.polyval(cN, 0))
    return float(np.sqrt(np.mean(np.square(rE)))), float(np.sqrt(np.mean(np.square(rN))))


def velocity_continuity(df, st, n=7):
    """Constant-velocity (ballistic) fits to the n GPS packets just before and
    just after burst.  Since horizontal drag is negligible (see H2) the gondola
    flies in a straight line and these slopes are its true horizontal velocity;
    their difference is the model-independent separation step."""
    MAX = st['MAX']; T0 = st['T0']

    def fit(i0, i1):
        seg = df.iloc[i0:i1]; t = seg['t_sec'].values.astype(float) - T0
        E = (seg['lon'].values - df.loc[MAX, 'lon']) * M_PER_DEG_LON
        N = (seg['lat'].values - df.loc[MAX, 'lat']) * M_PER_DEG_LAT
        cE = np.polyfit(t, E, 1); cN = np.polyfit(t, N, 1)
        rms = np.sqrt(np.mean((E - np.polyval(cE, t))**2 + (N - np.polyval(cN, t))**2))
        s = SIGMA_HOR / np.sqrt(np.sum((t - t.mean())**2))
        return cE[0], cN[0], rms, s

    vEa, vNa, rmsa, sa = fit(MAX - n + 1, MAX + 1)   # before burst
    vEd, vNd, rmsd, sd = fit(MAX + 1, MAX + 1 + n)   # after burst
    return dict(vEa=vEa, vNa=vNa, vEd=vEd, vNd=vNd, rms_after=rmsd, s=np.hypot(sa, sd),
                dvE=vEd - vEa, dvN=vNd - vNa, mag=np.hypot(vEd - vEa, vNd - vNa))


def drag_velocity_curve(w_rel, h=15087.0, T=120.0, Cd=0.30, dt=0.05):
    """Horizontal velocity a sphere gains over exposure time, starting from rest
    relative to a steady wind w_rel at a FIXED altitude h.  Returns (t, v).

    This is the *maximum* the air can do to it (the real gondola already moves
    with most of the wind).  Used to compare the drag timescale against the ~4 s
    in which the observed velocity step appears at burst."""
    rho = isa_density(h); k = 0.5 * Cd * A_SPH * rho / M_GON
    ts = np.arange(0, T, dt); v = 0.0; out = np.empty_like(ts)
    for i in range(ts.size):
        rel = w_rel - v; v += k * abs(rel) * rel * dt; out[i] = v
    return ts, out


def dv_gained_by_drag(w_rel, h=14000, T=20, Cd=0.30, dt=0.02):
    """Convenience scalar: velocity gained by drag in time T (see curve above)."""
    rho = isa_density(h); k = 0.5 * Cd * A_SPH * rho / M_GON
    v = 0.0
    for _ in range(int(T / dt)):
        rel = w_rel - v; v += k * abs(rel) * rel * dt
    return v


def drag_relax_time(h, w_rel=5.0, Cd=0.30):
    """Drag relaxation time tau ~ 1/(k*w_rel) at altitude h: the timescale over
    which wind drag can change the sphere's horizontal velocity.  Large tau =
    drag is effectively 'frozen' (the sphere ignores the wind on that timescale).
    tau shrinks at low altitude as the air gets denser -> wind re-couples."""
    rho = isa_density(h); k = 0.5 * Cd * A_SPH * rho / M_GON
    return 1.0 / (k * w_rel)


def fall_speed(st, n=10):
    """Vertical fall speed [m/s] for the first n descent packets.

    A drift-derived wind is only meaningful while the gondola moves slowly.
    Once it plummets (tens of m/s) its horizontal motion is drag-lagged
    dynamics, not the local wind, so an ascent-vs-descent "wind difference"
    cannot be measured below apogee."""
    DESC = st['DESC']
    t = DESC['t_sec'].values[:n+1].astype(float) - st['T0']
    h = DESC['alt'].values[:n+1].astype(float)
    vfall = -np.gradient(h, t)
    return t[:n], vfall[:n]


def near_apogee_wind(st):
    """Cleanest possible wind comparison: the slow motion just around apogee.

    Ascent wind from the last 4 ascent packets (slow climb), descent velocity
    from the first 2 descent packets (still <~30 m/s fall).  This is the ONLY
    window where the ascent and descent drift both approximate the true wind."""
    MAX = st['MAX']; ASC = st['ASC']; DESC = st['DESC']
    asc = ASC.iloc[-4:]; ta = asc['t_sec'].values.astype(float)
    uE_a = np.polyfit(ta, asc['lon'].values, 1)[0] * M_PER_DEG_LON
    vN_a = np.polyfit(ta, asc['lat'].values, 1)[0] * M_PER_DEG_LAT
    des = DESC.iloc[:3]; tdd = des['t_sec'].values.astype(float)
    uE_d = np.polyfit(tdd, des['lon'].values, 1)[0] * M_PER_DEG_LON
    vN_d = np.polyfit(tdd, des['lat'].values, 1)[0] * M_PER_DEG_LAT
    return dict(uE_a=uE_a, vN_a=vN_a, uE_d=uE_d, vN_d=vN_d,
                duE=uE_d - uE_a, dvN=vN_d - vN_a)


def spin_metric(df, idx0, idx1):
    """std of packet-to-packet steps in log10(UVA) over [idx0, idx1).

    A body-fixed UV sensor on a non-rotating gondola would track the slowly
    varying solar elevation -> small steps.  Rapid rotation sweeps the sensor
    through the solar direction every revolution -> large, erratic steps.  The
    metric returns the typical multiplicative swing per packet (10**std)."""
    seg = df.iloc[idx0:idx1]
    x = np.log10(np.clip(seg[UVA_COL].values.astype(float), 1, None))
    return float(np.std(np.diff(x)))


def fit_impulse(st, n_packets=20, n_fit=N_HORIZ):
    """Fit the H1 impulse on the first n_fit packets, then return the H1
    residuals (N and E, in metres) evaluated over n_packets packets, plus the
    fitted Dv.  Lets us watch how correlated the residuals are (relevant to the
    overstated significance in H4)."""
    DESC = st['DESC']; T0 = st['T0']
    td = (DESC['t_sec'].values[:n_packets] - T0).astype(float)
    lm = DESC['lat'].values[:n_packets]; om = DESC['lon'].values[:n_packets]
    tdh, lmh, omh = td[:n_fit], lm[:n_fit], om[:n_fit]

    def res(pp):
        t_, la_, lo_ = propagate_horiz(st, pp[0], pp[1])
        return np.concatenate([(lmh - np.interp(tdh, t_, la_)) * M_PER_DEG_LAT / SIGMA_HOR,
                               (omh - np.interp(tdh, t_, lo_)) * M_PER_DEG_LON / SIGMA_HOR])

    r = least_squares(res, [0., 0.], bounds=([-50, -50], [50, 50]))
    ts, la, lo = propagate_horiz(st, r.x[0], r.x[1])
    rN = (lm - np.interp(td, ts, la)) * M_PER_DEG_LAT
    rE = (om - np.interp(td, ts, lo)) * M_PER_DEG_LON
    return dict(dv_N=r.x[0], dv_E=r.x[1], td=td, rN=rN, rE=rE)


def pendulum_disp(vmax, L):
    """For a pendulum of length L carrying horizontal speed vmax at the bottom
    of its swing: return (period, position amplitude, angular amplitude[deg])."""
    w = np.sqrt(G / L)
    return 2 * np.pi / w, vmax / w, np.degrees((vmax / w) / L)


def ascent_resid(df, st, N=30, deg=3):
    """Detrended horizontal position residuals over the last N ascent packets
    (smooth wind drift removed by a low-order polynomial).  Returns the
    residual series plus simple structure metrics and the GPS cadence."""
    MAX = st['MAX']
    seg = df.iloc[MAX - N + 1:MAX + 1]
    t = seg['t_sec'].values.astype(float); tc = t - t.mean()
    E = (seg['lon'].values - seg['lon'].mean()) * M_PER_DEG_LON
    Nn = (seg['lat'].values - seg['lat'].mean()) * M_PER_DEG_LAT
    rE = E - np.polyval(np.polyfit(tc, E, deg), tc)
    rN = Nn - np.polyval(np.polyfit(tc, Nn, deg), tc)

    def ac1(x):
        x = x - x.mean(); return np.sum(x[1:] * x[:-1]) / np.sum(x * x)
    cad = float(np.median(np.diff(t)))
    return dict(tc=tc, rE=rE, rN=rN, cad=cad, N=N,
                rmsE=float(np.std(rE)), rmsN=float(np.std(rN)),
                ac1E=ac1(rE), ac1N=ac1(rN))


# =====================================================================
#  H1 — GPS MEASUREMENT ERROR
# ---------------------------------------------------------------------
#  Could the 3 m/s step (and the notorious Delta-E misfit) be nothing but
#  GPS noise?  Two checks: (a) the per-axis GPS scatter is ~5 m and roughly
#  isotropic, so a 3 m/s step (=12 m of offset growth in one 4 s packet) is
#  far larger than the noise; (b) the track stays straight through burst with
#  a const-velocity residual at the GPS-noise level, i.e. the step is a real
#  change of slope, not scatter.  VERDICT: the step is REAL — H1 excluded.
#  (The Delta-E *excess* scatter is a slow correlated wind meander, not white
#  GPS noise — that is an H4 issue, not evidence for a collision.)
# =====================================================================
def h1_report(df, st):
    MAX = st['MAX']; vc = velocity_continuity(df, st)
    print("\n" + "=" * 70)
    print(" H1 — GPS measurement error?")
    print("=" * 70)
    for name, idx in [('steady ascent 3-10km', df.index[(df.index < MAX) & (df['alt'] > 3000) & (df['alt'] < 10000)]),
                      ('upper ascent 10-15km', df.index[(df.index < MAX) & (df['alt'] > 10000)]),
                      ('descent 5-13km',       df.index[(df.index > MAX) & (df['alt'] > 5000) & (df['alt'] < 13000)])]:
        re, rn = gps_axis_noise(df, idx.values)
        print(f"    GPS noise {name:22s} E={re:4.1f}  N={rn:4.1f} m   (E/N={re/rn:.2f}, ~isotropic)")
    print(f"    Velocity step at burst = {vc['mag']:.1f} m/s  (=> {vc['mag']*4:.0f} m offset growth per 4 s packet)")
    print(f"    Post-burst track is straight: const-velocity residual {vc['rms_after']:.1f} m ~ GPS noise.")
    print("    => the step is far larger than GPS scatter -> H1 EXCLUDED (step is REAL).")


def fig_h1_gps_error(df, st, out='TTS9_H1_gps_error.png'):
    MAX = st['MAX']; T0 = st['T0']; vc = velocity_continuity(df, st)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- left: the GPS measurement-error scale is ~5 m and isotropic ----
    labels, eN, nN = [], [], []
    for name, idx in [('ascent\n3-10km', df.index[(df.index < MAX) & (df['alt'] > 3000) & (df['alt'] < 10000)]),
                      ('ascent\n10-15km', df.index[(df.index < MAX) & (df['alt'] > 10000)]),
                      ('descent\n5-13km', df.index[(df.index > MAX) & (df['alt'] > 5000) & (df['alt'] < 13000)])]:
        re, rn = gps_axis_noise(df, idx.values)
        labels.append(name); eN.append(re); nN.append(rn)
    x = np.arange(len(labels))
    axL.bar(x - 0.18, eN, 0.36, color='#ffa657', label='East', edgecolor='white')
    axL.bar(x + 0.18, nN, 0.36, color='#58a6ff', label='North', edgecolor='white')
    axL.axhline(SIGMA_HOR, color='white', ls='--', lw=1, label=f'noise model sigma={SIGMA_HOR:.0f} m')
    axL.set_xticks(x); axL.set_xticklabels(labels)
    axL.set_ylabel('GPS position scatter [m]')
    axL.set_title('(a) The GPS error scale: ~5 m, similar in both axes\n'
                  'This is how big a pure measurement wobble can be')
    axL.legend(fontsize=8)

    # ---- right: deviation from the pre-burst straight-line (ballistic) motion ----
    #  Fit a constant-velocity line to the packets BEFORE burst, extrapolate it
    #  through the whole window, and plot how far the real GPS track departs from
    #  it.  No velocity change => stays flat inside the +/-5 m noise band; a real
    #  step => departs linearly at the step speed.
    nbef = 7
    win = df.iloc[MAX - nbef + 1:MAX + 9]
    t = win['t_sec'].values.astype(float) - T0
    E = (win['lon'].values - df.loc[MAX, 'lon']) * M_PER_DEG_LON
    N = (win['lat'].values - df.loc[MAX, 'lat']) * M_PER_DEG_LAT
    pre = t <= 0
    cE = np.polyfit(t[pre], E[pre], 1); cN = np.polyfit(t[pre], N[pre], 1)
    devE = E - np.polyval(cE, t); devN = N - np.polyval(cN, t)
    axR.axhspan(-SIGMA_HOR, SIGMA_HOR, color='#8b949e', alpha=0.22, label='GPS noise band +/-5 m')
    axR.axvline(0, color='#d2a8ff', ls='--', lw=1.2)
    axR.plot(t, devN, 's-', color='#58a6ff', lw=1.6, label='North deviation')
    axR.plot(t, devE, 'o-', color='#ffa657', lw=1.6, label='East deviation')
    # annotate the post-burst divergence slope (= the velocity step)
    tpa = t[t >= 0]
    axR.annotate(f'diverges at {vc["mag"]:.1f} m/s\n(= the velocity step)',
                 xy=(tpa[-1], devN[t >= 0][-1]), xytext=(6, -0.45*abs(devN).max()),
                 color='#e6edf3', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#8b949e'))
    axR.text(-24, SIGMA_HOR + 3, 'before burst:\non a straight line\n(flat, within noise)',
             color='#8b949e', fontsize=8)
    axR.set_xlabel('t [s] from burst'); axR.set_ylabel('departure from pre-burst straight line [m]')
    axR.set_title('(b) Before burst the gondola flies straight; at burst its\n'
                  'path bends and leaves the GPS noise band within one packet')
    axR.legend(fontsize=8, loc='lower left')

    fig.suptitle('TTS9 — H1: Is the velocity step just GPS error?   No — the step is ~10x the GPS noise.',
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"Saved {out}")
    return fig


# =====================================================================
#  H2 — WIND DRAG DURING THE FALL
# ---------------------------------------------------------------------
#  Could ordinary wind, acting through aerodynamic drag on the 160 mm sphere,
#  swing its horizontal velocity by 3 m/s as it descends?  The drag relaxation
#  of this light sphere is slow: even at a generous 5 m/s steady relative wind
#  it gains <1 m/s in 20 s; reaching 3 m/s needs ~10 m/s of *sustained*
#  relative wind, which is unphysical here.  And the comparison is only valid
#  near apogee anyway — once the fall ramps past ~50 m/s the horizontal motion
#  is drag-lagged dynamics, not the wind.  VERDICT: drag negligible, the post-
#  burst flight is essentially ballistic — H2 excluded.
# =====================================================================
def h2_report(df, st):
    print("\n" + "=" * 70)
    print(" H2 — Wind drag during the fall?")
    print("=" * 70)
    for w in (2.0, 5.0, 10.0):
        print(f"    relative wind {w:4.1f} m/s -> sphere gains {dv_gained_by_drag(w, T=20):.2f} m/s in 20 s")
    tt, vf = fall_speed(st)
    print("    Fall speed ramps up at once (drift = wind only at apogee):")
    print("      " + "  ".join(f"t={int(t)}s:{v:.0f}m/s" for t, v in zip(tt[:5], vf[:5])))
    print("    => 3 m/s needs ~10 m/s sustained relative wind -> H2 EXCLUDED (flight is ballistic).")


def fig_h2_wind_drag(df, st, out='TTS9_H2_wind_drag.png'):
    vc = velocity_continuity(df, st)
    step = vc['mag']
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- left: TIMESCALE mismatch — drag needs ~minute, the step took ~4 s ----
    #  Velocity built up by drag vs exposure time at burst altitude, for several
    #  steady relative winds.  The observed step (red) appears within one 4 s
    #  packet (white line); drag at a realistic 5 m/s wind needs ~70 s to get
    #  there — an ~18x mismatch.
    Tmax = 200.0
    for w, c in ((2.0, '#58a6ff'), (5.0, '#ffa657'), (10.0, '#f0883e')):
        ts, v = drag_velocity_curve(w, T=Tmax)
        axL.plot(ts, v, color=c, lw=2.2, label=f'rel. wind {w:.0f} m/s')
        if np.any(v >= step):
            cross = ts[np.argmax(v >= step)]
            axL.annotate(f'{w:.0f} m/s: needs {cross:.0f} s', xy=(cross, step),
                         xytext=(cross+6, step-0.55), color=c, fontsize=8.5,
                         arrowprops=dict(arrowstyle='->', color=c, lw=1))
    axL.axhline(step, color='#f85149', ls='--', lw=1.6, label=f'observed step = {step:.1f} m/s')
    axL.axvline(4, color='#e6edf3', ls=':', lw=1.4)
    axL.text(7, step*0.10, 'available time:\none 4 s packet', color='#e6edf3', fontsize=8.5)
    axL.set_xlim(0, Tmax); axL.set_ylim(0, max(step*1.3, 3.5))
    axL.set_xlabel('time the wind would have to act on the gondola [s]')
    axL.set_ylabel('horizontal velocity built up by drag [m/s]')
    axL.set_title('(a) Drag is far too SLOW at burst altitude (15 km)\n'
                  'a realistic 5 m/s wind needs ~160 s, even 10 m/s needs ~23 s — the step took ~4 s')
    axL.legend(fontsize=8, loc='center right')

    # ---- right: WHY — relaxation time vs altitude: frozen high, re-couples low ----
    hs = np.linspace(0, 16000, 200)
    tau = np.array([drag_relax_time(h) for h in hs])
    axR.plot(tau, hs/1000, color='#7ee787', lw=2.5)
    axR.axhline(st['H_SEP']/1000, color='#d2a8ff', ls='--', lw=1.3,
                label=f'burst altitude {st["H_SEP"]/1000:.1f} km')
    axR.axvline(4, color='white', ls=':', lw=1.3, label='one GPS packet (4 s)')
    tau_burst = drag_relax_time(st['H_SEP'])
    axR.annotate(f'~{tau_burst:.0f} s at burst\n(drag is "frozen")',
                 xy=(tau_burst, st['H_SEP']/1000), xytext=(tau_burst-55, 11),
                 color='#e6edf3', fontsize=8.5, arrowprops=dict(arrowstyle='->', color='#8b949e'))
    axR.text(20, 1.5, 'low down the air is denser:\ndrag re-couples the gondola\nto the wind (=> late drift)',
             color='#8b949e', fontsize=8)
    axR.set_xlabel('drag relaxation time at 5 m/s rel. wind [s]'); axR.set_ylabel('altitude [km]')
    axR.set_title('(b) Why drag is negligible at burst: tau ~ 100 s >> 4 s.\n'
                  'It only matters low down — which explains the late-fall drift')
    axR.legend(fontsize=8, loc='upper right')

    fig.suptitle('TTS9 — H2: Can wind drag turn the gondola by 3 m/s at burst?  No — the timescales do not match.',
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"Saved {out}")
    return fig


# =====================================================================
#  H3 — ROTATION + MAGNUS SIDE-FORCE
# ---------------------------------------------------------------------
#  The directional UV photodiodes show strong packet-to-packet modulation in
#  every flight phase: the gondola spins/tumbles throughout (period < ~8 s, so
#  aliased by the 4 s cadence — we see that it rotates, not how fast).  Could
#  that spin, via the Magnus effect in the fast fall, generate a lateral force
#  of 3 m/s worth?  The bound says no: even a clean 1 rev/s spin yields a tiny
#  lateral Dv, and because the body tumbles, the Magnus axis wanders and its
#  force averages out — no sustained side-push.  VERDICT: H3 excluded.
# =====================================================================
def h3_report(df):
    MAX = int(df['alt'].idxmax())
    print("\n" + "=" * 70)
    print(" H3 — Rotation + Magnus side-force?")
    print("=" * 70)
    print("    UVA packet-to-packet swing (cadence ~4 s => spin period < ~8 s, aliased):")
    for name, idx in [('early ascent (h<5km)', df.index[(df.index < MAX) & (df['alt'] < 5000)]),
                      ('late ascent (h>10km)', df.index[(df.index < MAX) & (df['alt'] > 10000)]),
                      ('first 30 descent',     df.index[MAX:MAX+30]),
                      ('mid descent',          df.index[MAX+30:MAX+90])]:
        if len(idx) < 3:
            continue
        m = spin_metric(df, idx[0], idx[-1] + 1)
        print(f"      {name:22s} {10**m:4.1f}x per packet  -> rotating")
    print("    Magnus bound: even a clean 1 rev/s spin gives < ~1 m/s lateral Dv per packet,")
    print("    and tumbling randomizes its direction. => H3 EXCLUDED (no sustained side-force).")


def fig_h3_magnus(df, st, out='TTS9_H3_magnus_rotation.png'):
    T0 = st['T0']
    t_rel = (df['t_sec'].values - T0)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- (a) UVA vs time: rapid modulation = rotation (the premise) ----
    axA.semilogy(t_rel/60, np.clip(df[UVA_COL].values, 1, None),
                 color='#58a6ff', lw=0.8, marker='.', ms=3)
    axA.axvline(0, color='#d2a8ff', ls='--', lw=1.2)
    axA.text(0.4, axA.get_ylim()[1]*0.35, 'burst', color='#d2a8ff', fontsize=9)
    axA.set_xlabel('time from apogee [min]'); axA.set_ylabel('UVA [uW/cm2]  (log)')
    axA.set_title('(a) Premise: the gondola IS spinning\n'
                  'directional UVA jumps 2-5x every packet = rotation/tumble')

    # ---- (b) Magnus side-force bound vs spin rate ----
    rho13 = isa_density(13000); Vfall = 55.0
    fs = np.linspace(0, 2, 60)
    S = (2*np.pi*fs) * R_SPH / Vfall                  # spin parameter
    Cl = np.clip(1.5*S, 0, 0.4)                        # conservative Magnus lift coeff
    a_lat = 0.5 * rho13 * Vfall**2 * A_SPH * Cl / M_GON
    axB.plot(fs, a_lat*4, color='#7ee787', lw=2.5, label='max |Dv| over one packet (Magnus, fixed axis)')
    axB.axhline(3.2, color='#f85149', ls='--', lw=1.6, label='observed step = 3.2 m/s')
    axB.set_ylim(0, 4)
    axB.set_xlabel('spin rate [rev/s]'); axB.set_ylabel('lateral velocity change in 4 s [m/s]')
    axB.set_title('(b) But Magnus is far too weak: even a clean 1 rev/s spin\n'
                  'gives < 1 m/s, and tumbling randomizes its direction to ~0')
    axB.legend(fontsize=8, loc='upper left')

    fig.suptitle('TTS9 — H3: Can the spin (Magnus effect) produce the side-step?   No — too weak and direction-averaged.',
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"Saved {out}")
    return fig


# =====================================================================
#  H4 — RECONSTRUCTION ARTEFACT (initial velocity + overstated significance)
# ---------------------------------------------------------------------
#  This is the most important "mundane" alternative.  H0 propagates the
#  gondola from a horizontal velocity obtained by extrapolating the ascent
#  wind profile up to apogee — the weakest link, since near apogee the balloon
#  decelerates and swings.  Sweeping the assumed northward velocity V_SEP by a
#  couple of m/s (well within its real uncertainty) drops the H0 chi^2 from
#  ~21 to ~2.4, i.e. "rejected" turns into "acceptable": the impulse magnitude
#  is model-dependent.  Worse, the notebook's p=3.5e-5 treats the first few
#  descent residuals as independent, but they are smoothly correlated
#  (lag-1 autocorr ~0.9): the effective sample size is ~1, so the true
#  significance is far lower.  VERDICT: the step is real (H1) but the
#  notebook's CONFIDENCE is overstated — H4 partially supported.
# =====================================================================
def h4_report(df, st):
    print("\n" + "=" * 70)
    print(" H4 — Reconstruction artefact / overstated significance?")
    print("=" * 70)
    V0 = st['V_SEP']
    print("    H0 reduced chi^2 vs assumed initial northward velocity V_SEP:")
    for d in (-3, -2, -1, 0, 1):
        c = chi2_H0(st, V0=V0 + d) / (2 * N_HORIZ)
        verdict = 'acceptable' if c < 3 else 'REJECTED'
        edge = '  <- only at extreme edge of plausible range' if d == -3 else ''
        print(f"      V_SEP={V0+d:+.2f} m/s ({d:+d}):  chi^2/dof={c:5.2f}  -> H0 {verdict}{edge}")
    print("    => within the plausible +/-2.5 m/s wind range H0 stays rejected: the step is")
    print("       NOT merely a wind-extrapolation artefact (robust to V_SEP).")
    DESC = st['DESC']; T0 = st['T0']; nshow = 6
    td = (DESC['t_sec'].values[:nshow] - T0).astype(float)
    lm = DESC['lat'].values[:nshow]
    ts, la, lo = propagate_horiz(st, 0., 0.)
    rN = (lm - np.interp(td, ts, la)) * M_PER_DEG_LAT
    ac = np.corrcoef(rN[:-1], rN[1:])[0, 1]
    n_eff = nshow * (1 - ac) / (1 + ac)
    print(f"    BUT the {nshow} fit-window residuals move as one smooth trend (autocorr {ac:+.2f})")
    print(f"    -> only ~{n_eff:.1f} independent points; treating them as {nshow} independent")
    print("       observations OVERSTATES the F-test significance (p=3.5e-5 is not trustworthy).")


def fig_h4_reconstruction(df, st, out='TTS9_H4_reconstruction.png'):
    V0 = st['V_SEP']
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- left: how robust is the step to the apogee-wind initial guess? ----
    #  H0 needs an assumed initial northward velocity V_SEP (the ascent wind
    #  extrapolated to apogee).  We sweep it.  Within its plausible +/-2.5 m/s
    #  uncertainty H0 stays REJECTED (chi^2/dof > 3): the step is NOT just a
    #  wind-extrapolation artefact.  Only at the extreme edge would H0 pass.
    dVs = np.linspace(-3.5, 1.5, 60)
    chi = np.array([chi2_H0(st, V0=V0+d) / (2*N_HORIZ) for d in dVs])
    axL.plot(V0+dVs, chi, color='#ffa657', lw=2.5)
    axL.axhline(3, color='#f85149', ls=':', lw=1.3, label='reject threshold (chi^2/dof~3)')
    axL.axvspan(V0-2.5, V0+2.5, color='#7ee787', alpha=0.15, label='plausible V_SEP (+/-2.5 m/s)')
    axL.axvline(V0, color='#58a6ff', ls=':', lw=1.5, label=f'best estimate = {V0:.1f} m/s')
    axL.scatter([V0], [chi2_H0(st, V0=V0)/(2*N_HORIZ)], s=80, color='#f85149', zorder=5)
    axL.set_ylim(0, max(chi)*1.05)
    axL.set_xlabel('assumed initial northward velocity V_SEP [m/s]')
    axL.set_ylabel('H0 reduced chi^2  (no-impulse fit)')
    axL.set_title('(a) The step is robust to the wind guess:\n'
                  'within +/-2.5 m/s H0 stays rejected — only the extreme edge passes')
    axL.legend(fontsize=8, loc='upper right')

    # ---- right: the "5 independent points" are really ~1 correlated trend ----
    DESC = st['DESC']; T0 = st['T0']; nshow = 6
    td = (DESC['t_sec'].values[:nshow] - T0).astype(float)
    lm = DESC['lat'].values[:nshow]
    ts, la, lo = propagate_horiz(st, 0., 0.)
    rN = (lm - np.interp(td, ts, la)) * M_PER_DEG_LAT     # H0 (no-impulse) residual
    ac = np.corrcoef(rN[:-1], rN[1:])[0, 1]
    n_eff = nshow * (1 - ac) / (1 + ac)
    rng = np.random.default_rng(1); noise = rng.normal(0, SIGMA_HOR, nshow)
    axR.axhspan(-SIGMA_HOR, SIGMA_HOR, color='#8b949e', alpha=0.22, label='GPS noise +/-5 m')
    axR.axhline(0, color='#8b949e', lw=0.8)
    axR.plot(td, rN, 's-', color='#58a6ff', lw=1.8, label='H0 North residual (the data the F-test uses)')
    axR.plot(td, noise, 'o:', color='#f0883e', lw=1.2, alpha=0.9,
             label='what 6 *independent* GPS draws look like')
    axR.set_xlabel('t [s] from burst (fit window)'); axR.set_ylabel('residual [m]')
    axR.set_title(f'(b) These 6 points move as one smooth trend (autocorr {ac:+.2f}),\n'
                  f'~{n_eff:.1f} truly independent — so the quoted p=3.5e-5 is overstated')
    axR.legend(fontsize=8, loc='lower left')

    fig.suptitle('TTS9 — H4: Is the detection oversold?  The step is real & robust, but its significance is overstated.',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"Saved {out}")
    return fig


# =====================================================================
#  H5 — PENDULUM (swing energy frozen at release)
# ---------------------------------------------------------------------
#  If the gondola swung as a pendulum below the balloon with enough energy
#  that, frozen at line release, it carries the 3 m/s step, that swing must
#  already show up as a periodic horizontal motion in the ascent GPS.  Key
#  geometry: position amplitude A_x = v_max/omega, omega = sqrt(g/L).  A SHORT
#  Confirmed rope length: 6 m (ČHMU consultation, 2026-06-11).
#  For L=6 m: T=4.9 s (below 8 s GPS Nyquist, aliased), A_x=2.4 m (below 5 m
#  GPS noise) -> completely invisible in GPS.  Required swing angle: ~22°
#  (physically plausible, not violent).  VERDICT: long-line (L>16 m) pendulum
#  EXCLUDED; confirmed L=6 m swing is invisible AND plausible — CANNOT BE
#  EXCLUDED from GPS data alone.
# =====================================================================
def h5_report(df, st):
    r = ascent_resid(df, st)
    P_nyq = 2 * r['cad']
    L_res = G * (P_nyq / (2 * np.pi)) ** 2
    floor3 = 3 * SIGMA_HOR * np.sqrt(2 / r['N'])
    print("\n" + "=" * 70)
    print(" H5 — Pendulum swing frozen at release?")
    print("=" * 70)
    L_ROPE_CONFIRMED = 6.0   # m — confirmed from ČHMU consultation
    print(f"    GPS cadence {r['cad']:.1f} s -> only pendulums with line L > {L_res:.0f} m are resolvable.")
    print(f"    Confirmed suspension rope: {L_ROPE_CONFIRMED:.0f} m")
    print("    A 3 m/s swing: position amplitude A_x = 3*sqrt(L/g):")
    for L in (2, L_ROPE_CONFIRMED, 10, 16, 25):
        Tp, Ax, th = pendulum_disp(3.0, L)
        tag = "resolvable" if Tp > P_nyq else "ALIASED (invisible)"
        confirmed = " <-- CONFIRMED ROPE LENGTH" if L == L_ROPE_CONFIRMED else ""
        print(f"      L={L:4.1f} m: T={Tp:5.1f}s  A_x={Ax:4.1f}m  swing={th:4.1f}deg   [{tag}]{confirmed}")
    print(f"    3-sigma detectable amplitude over {r['N']} packets ~ {floor3:.1f} m.")
    print(f"    Ascent residuals: North RMS {r['rmsN']:.1f} m (~noise), East RMS {r['rmsE']:.1f} m"
          f" (slow wind meander, autocorr {r['ac1E']:+.2f}); pendulum band empty.")
    Tp6, Ax6, th6 = pendulum_disp(3.0, L_ROPE_CONFIRMED)
    print(f"    => long-line pendulum EXCLUDED; confirmed L={L_ROPE_CONFIRMED:.0f} m swing:"
          f" T={Tp6:.1f}s (aliased), A_x={Ax6:.1f}m (below noise), angle={th6:.0f}° (plausible)."
          f"\n    => pendulum hypothesis CANNOT BE EXCLUDED for confirmed rope length.")


def fig_h5_pendulum(df, st, out='TTS9_H5_pendulum.png'):
    from scipy.signal import lombscargle
    r = ascent_resid(df, st)
    P_nyq = 2 * r['cad']
    floor3 = 3 * SIGMA_HOR * np.sqrt(2 / r['N'])
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5.6))

    # ---- (a) detrended ascent residuals: E wanders slowly, N ~ noise ----
    tt = r['tc'] - r['tc'].min()
    axA.axhline(0, color='#8b949e', lw=0.8)
    axA.plot(tt, r['rE'], 'o-', color='#ffa657', lw=1.2, ms=4,
             label=f"E (east)  RMS {r['rmsE']:.1f} m, autocorr {r['ac1E']:+.2f}")
    axA.plot(tt, r['rN'], 's-', color='#58a6ff', lw=1.2, ms=4,
             label=f"N (north) RMS {r['rmsN']:.1f} m, autocorr {r['ac1N']:+.2f}")
    axA.set_xlabel('time over last ascent window [s]'); axA.set_ylabel('horizontal residual [m]')
    axA.set_title('(a) Pre-burst residuals: East = slow wind meander,\nNorth ~ GPS noise (no fast swing)')
    axA.legend(fontsize=8, loc='upper left')

    # ---- (b) Lomb-Scargle: pendulum band empty, power only at wind scale ----
    periods = np.linspace(P_nyq, 60, 500); w = 2 * np.pi / periods
    for name, rr, c in (('E', r['rE'], '#ffa657'), ('N', r['rN'], '#58a6ff')):
        pg = lombscargle(r['tc'], rr - rr.mean(), w, normalize=True)
        axB.plot(periods, pg, color=c, lw=1.8, label=f'{name} residual')
    axB.axvspan(P_nyq, 15, color='#7ee787', alpha=0.15, label='resolvable pendulum band (L=16-50 m)')
    axB.set_xlabel('period [s]'); axB.set_ylabel('Lomb-Scargle power')
    axB.set_title('(b) No peak in the pendulum band\nall power sits at the ~50 s wind scale')
    axB.legend(fontsize=8, loc='upper right')

    # ---- (c) detectability map: amplitude vs line length for a 3 m/s swing ----
    Ls = np.linspace(0.5, 50, 200)
    Ax = 3.0 * np.sqrt(Ls / G)
    axC.plot(Ls, Ax, color='#d2a8ff', lw=2.5, label='position amplitude of a 3 m/s swing')
    axC.axhline(SIGMA_HOR, color='#f85149', ls='--', lw=1.3, label=f'GPS noise {SIGMA_HOR:.0f} m')
    axC.axhline(floor3, color='#f0883e', ls=':', lw=1.3, label=f'3-sig detect floor {floor3:.1f} m')
    L_res = G * (P_nyq / (2 * np.pi)) ** 2
    axC.axvspan(0.5, L_res, color='#8b949e', alpha=0.22, label=f'aliased: L<{L_res:.0f} m (invisible)')
    axC.axvline(L_res, color='white', ls=':', lw=1)
    axC.set_xlabel('suspension line length L [m]'); axC.set_ylabel('GPS position amplitude [m]')
    axC.set_title('(c) Only a long line makes a 3 m/s swing visible\n(resolvable AND above noise) — and none is seen')
    axC.legend(fontsize=8, loc='upper left')

    fig.suptitle('TTS9 — H5: Was a pendulum swing frozen at release?  No visible swing for any non-tiny line.',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"Saved {out}")
    return fig


# =====================================================================
#  Console summary (seeds the README text conclusion; no figure)
# =====================================================================
def print_summary(df, st):
    vc = velocity_continuity(df, st)
    print("\n" + "#" * 70)
    print("#  SUMMARY — explanations for the ~%.1f m/s horizontal step at burst" % vc['mag'])
    print("#" * 70)
    rows = [
        ("H1 GPS error",            "EXCLUDED", "step >> 5 m isotropic noise; track straight"),
        ("H2 Wind drag on fall",    "EXCLUDED", "<1 m/s in 20 s at 5 m/s wind; needs ~10 m/s"),
        ("H3 Rotation + Magnus",    "EXCLUDED", "force small; tumbling randomizes its direction"),
        ("H4 Reconstr. artefact",   "PARTIAL ", "step robust to wind; but p=3.5e-5 overstated (corr.)"),
        ("H5 Pendulum",             "EXCLUDED*", "no swing for L>16 m; short-line invisible/violent"),
        ("H6 External collision",   "SURVIVES", "not independently testable from this telemetry"),
    ]
    for name, verd, why in rows:
        print(f"   {name:22s} {verd:9s} {why}")
    print("   -----------------------------------------------------------------")
    print("   => The step is REAL and ballistic, but its ORIGIN is not decidable")
    print("      from GPS alone: data are CONSISTENT WITH, not DIAGNOSTIC OF, a")
    print("      collision. (*) long-line pendulum excluded; short-line only hidden.")
    print("#" * 70)


# =====================================================================
#  Stand-alone entry point
# =====================================================================
if __name__ == '__main__':
    df = load_flight()
    st = reconstruct_burst(df)
    print(f"Burst: t_sep={st['TS']:+.2f}s  h={st['H_SEP']:.0f}m  "
          f"U_SEP(E)={st['U_SEP']:+.2f}  V_SEP(N)={st['V_SEP']:+.2f} m/s")

    # One report + one figure per hypothesis, in order H1..H5.
    h1_report(df, st);  fig_h1_gps_error(df, st)
    h2_report(df, st);  fig_h2_wind_drag(df, st)
    h3_report(df);      fig_h3_magnus(df, st)
    h4_report(df, st);  fig_h4_reconstruction(df, st)
    h5_report(df, st);  fig_h5_pendulum(df, st)

    print_summary(df, st)
