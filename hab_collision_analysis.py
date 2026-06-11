"""
HAB Collision Analysis — standalone script
Extracts gondola separation point and tests H0/H1 (natural burst vs. external impulse).

Configure the DATA section below and run:
    python hab_collision_analysis.py
    python hab_collision_analysis.py --flight DATA.TXT --date 2026-05-15
    python hab_collision_analysis.py --flight DATA+SENSORS-flight.xlsx --date 2026-04-29 --sounding EZM00011520-data.txt.zip
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.stats import f as fdist
import warnings, zipfile
warnings.filterwarnings('ignore')

# ── DATA (defaults, overridable via CLI) ────────────────────────────────────
FLIGHT_XLSX   = 'DATA+SENSORS-flight.xlsx'
SOUNDING_ZIP  = 'EZM00011520-data-beg2025.txt.zip'
SOUNDING_DATE = '2026 04 29'   # YYYY MM DD
SOUNDING_HOUR = 12             # UTC

# Gondola / balloon system parameters (verified with ČHMU, 2026-06-11)
M_GON  = 0.410         # kg  — total payload + sphere (Kaymont calculator: 410 g)
D_SPH  = 0.160         # m   — polystyrene sphere diameter
M_BAL  = 0.800         # kg  — balloon latex (Kaymont-800)
V_GAS  = 1.85          # m³  — hydrogen fill at launch
L_ROPE = 6.0           # m   — suspension rope length

# Fit parameters
N_ASC_FIT  = 30        # last N ascent packets used for polynomial fit
N_DESC_FIT = 25        # first N descent packets used for kinematic fit
N_HORIZ    = 5         # first N descent packets used for horizontal test

SIGMA_H   = 12.0       # GPS vertical noise [m]
SIGMA_HOR =  5.0       # GPS horizontal noise [m]

OUTPUT_PNG = 'HAB_final_analysis.png'

# ── PLOT STYLE ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',   'axes.labelcolor': '#e6edf3',
    'axes.grid': True,             'grid.color': '#30363d', 'grid.alpha': 0.55,
    'text.color': '#e6edf3',       'xtick.color': '#e6edf3', 'ytick.color': '#e6edf3',
    'legend.facecolor': '#161b22', 'legend.edgecolor': '#30363d',
    'figure.dpi': 130,
})
COLORS = dict(asc='#58a6ff', desc='#f85149', burst='#d2a8ff',
              snd='#ff9500', h0='#f85149', h1='#ffa657', direct='#7ee787')


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

A_SPH = np.pi * (D_SPH / 2) ** 2

LAT_C          = 49.70
M_PER_DEG_LAT  = 111320.0
M_PER_DEG_LON  = 111320.0 * np.cos(np.radians(LAT_C))


def isa_density(h):
    h = max(float(h), 0); g = 9.80665; R = 287.05287
    if h <= 11000:
        T = 288.15 - 0.0065 * h
        p = 101325 * (T / 288.15) ** (g / (R * 0.0065))
    else:
        T = 216.65; p = 22632 * np.exp(-g * (h - 11000) / (R * T))
    return p / (R * T)


def isa_pressure_hPa(h):
    g = 9.80665; R = 287.05287
    if h <= 11000:
        T = 288.15 - 0.0065 * h
        return 1013.25 * (T / 288.15) ** (g / (R * 0.0065))
    return 226.32 * np.exp(-g * (h - 11000) / (R * 216.65))


def geom_to_pa_ft(h):
    p = isa_pressure_hPa(h); g = 9.80665; R = 287.05287
    T0 = 288.15; L = 0.0065; p0 = 1013.25
    if p > 226.32:
        pa_m = (T0 / L) * (1 - (p / p0) ** (R * L / g))
    else:
        pa_m = 11000 + R * 216.65 / g * np.log(226.32 / p)
    return pa_m * 3.28084


def isa_height_from_pressure(p_hPa):
    g = 9.80665; R = 287.05287; T0 = 288.15; L = 0.0065; p0 = 1013.25
    if p_hPa > 226.32:
        return (T0 / L) * (1 - (p_hPa / p0) ** (R * L / g))
    return 11000 + R * 216.65 / g * np.log(226.32 / p_hPa)


def to_sec(s):
    if hasattr(s, 'hour'):
        return s.hour * 3600 + s.minute * 60 + s.second
    h, m, sc = str(s).split(':')
    return int(h) * 3600 + int(m) * 60 + int(sc)


# ═══════════════════════════════════════════════════════════════════════════
# 1  LOAD FLIGHT DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_flight(path):
    """Load flight data from Excel or TTS11 text log, auto-detected by extension."""
    if path.lower().endswith('.txt'):
        return _load_flight_txt(path)
    return _load_flight_xlsx(path)


def _load_flight_xlsx(path):
    df_raw = pd.read_excel(path)
    df_raw = df_raw.rename(columns={
        'čas[UTC]': 'time[UTC]',
        'Šířka[°]':  'lat[deg]',
        'Délka[°]':  'lon[deg]',
        'výška[m]':  'alt[m]',
    })
    df_raw['lat[deg]'] = pd.to_numeric(df_raw['lat[deg]'], errors='coerce')
    df_raw['lon[deg]'] = pd.to_numeric(df_raw['lon[deg]'], errors='coerce')
    df = df_raw[df_raw['alt[m]'].notna() & (df_raw['lat[deg]'] > 10)].copy()
    df = df.reset_index(drop=True)
    df['t_sec'] = df['time[UTC]'].apply(to_sec)
    df['t_min'] = (df['t_sec'] - df['t_sec'].iloc[0]) / 60.0
    return df


def _load_flight_txt(path):
    """Parse TTS11 text log: $$$$TTS11,seq,HH:MM:SS,lat,lon,alt_m,..."""
    records = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('$$$$'):
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            try:
                time_str = parts[2].strip()
                lat  = float(parts[3])
                lon  = float(parts[4])
                alt  = float(parts[5])
                if lat < 10:
                    continue
                records.append({'time[UTC]': time_str, 'lat[deg]': lat,
                                'lon[deg]': lon, 'alt[m]': alt})
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(records).reset_index(drop=True)
    df['t_sec'] = df['time[UTC]'].apply(to_sec)
    df['t_min'] = (df['t_sec'] - df['t_sec'].iloc[0]) / 60.0
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2  KINEMATIC BURST RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def sim_descent(k, h0, vh0, t_start, t_eval, dt=0.02):
    h = h0; v = vh0; t = t_start; ht = [h]; tt = [t]
    while h > 0 and t < t_eval[-1] + 5:
        rho = isa_density(h)
        a = 9.80665 - k * rho * v ** 2 if v >= 0 else 9.80665 + k * rho * v ** 2
        v += a * dt; h -= v * dt; t += dt; ht.append(h); tt.append(t)
    return np.interp(t_eval, tt, ht)


def reconstruct_burst(ASC, DESC, T0):
    af = ASC.tail(N_ASC_FIT)
    ta = (af['t_sec'].values - T0).astype(float)
    p_h   = np.polyfit(ta, af['alt[m]'].values.astype(float),  2)
    p_lat = np.polyfit(ta, af['lat[deg]'].values.astype(float), 2)
    p_lon = np.polyfit(ta, af['lon[deg]'].values.astype(float), 2)
    dp_h  = np.polyder(p_h)

    df_d  = DESC.head(N_DESC_FIT)
    td_fit = (df_d['t_sec'].values - T0).astype(float)
    hd_fit = df_d['alt[m]'].values.astype(float)

    def fit_descent_tsep(tsep):
        # Only use packets that are after the burst — pre-burst packets would
        # compare a post-apogee polynomial extrapolation against already-falling
        # GPS data, artificially pulling tsep to unrealistically large values.
        mask = td_fit >= tsep
        if mask.sum() < 5:
            return None, 1e9
        td_post = td_fit[mask]
        hd_post = hd_fit[mask]
        h0s  = float(np.polyval(p_h,   tsep))
        vh0s = -float(np.polyval(dp_h, tsep))
        def res(params):
            k = params[0]
            if not (0.005 < k < 0.5):
                return np.ones(len(td_post)) * 1e6
            return (sim_descent(k, h0s, vh0s, tsep, td_post) - hd_post) / SIGMA_H
        r = least_squares(res, [0.03], bounds=([0.005], [0.5]),
                          method='trf', ftol=1e-12, xtol=1e-12)
        return r, np.sqrt(np.mean(r.fun ** 2)) * SIGMA_H

    td_all = (DESC['t_sec'].values - T0).astype(float)
    tsep_scan = np.linspace(-6, 10, 160)
    rms_scan  = [fit_descent_tsep(ts)[1] for ts in tsep_scan]
    rms_scan  = np.array(rms_scan)

    best_i    = np.argmin(rms_scan)
    TS_OPT    = tsep_scan[best_i]
    r_best, _ = fit_descent_tsep(TS_OPT)
    K_VERT    = float(r_best.x[0])

    VH0_SEP = -float(np.polyval(dp_h, TS_OPT))

    # Burst position: extrapolate the PRE-BURST (ascending/balloon) trajectory to
    # TS_OPT using only packets confirmed to be before the burst.  This separates
    # the ascending and descending trajectories so any velocity discontinuity at
    # burst can be detected by the hypothesis test.  Mixing a pre-burst and a
    # post-burst GPS packet in a simple interpolation would partially absorb the
    # velocity discontinuity into the starting-position estimate.
    pre_mask = td_all < TS_OPT
    pre_idx  = np.where(pre_mask)[0]

    if len(pre_idx) >= 2:
        # Two or more confirmed pre-burst GPS packets: extrapolate their trend.
        i1, i2 = pre_idx[-2], pre_idx[-1]
        t1, t2 = td_all[i1], td_all[i2]
        def _extrap(col):
            v1 = float(DESC[col].values[i1])
            v2 = float(DESC[col].values[i2])
            return v2 + (v2 - v1) / (t2 - t1) * (TS_OPT - t2)
        H_SEP   = _extrap('alt[m]')
        LAT_SEP = _extrap('lat[deg]')
        LON_SEP = _extrap('lon[deg]')
    else:
        # TS_OPT is before (or at) the first DESC GPS packet — fall back to the
        # ascending polynomial, which is reliable within its fitting range.
        H_SEP   = float(np.polyval(p_h,   TS_OPT))
        LAT_SEP = float(np.polyval(p_lat, TS_OPT))
        LON_SEP = float(np.polyval(p_lon, TS_OPT))

    t_burst_abs = T0 + TS_OPT
    hh = int(t_burst_abs // 3600)
    mm = int((t_burst_abs % 3600) // 60)
    ss = int(t_burst_abs % 60)

    print(f"=== KINEMATIC BURST RECONSTRUCTION ===")
    print(f"  t_sep  = {TS_OPT:+.2f} s  (relative to GPS apogee)")
    print(f"  Time:    {hh:02d}:{mm:02d}:{ss:02d} UTC")
    print(f"  Alt:     {H_SEP:.1f} m")
    print(f"  Lat:     {LAT_SEP:.6f} N")
    print(f"  Lon:     {LON_SEP:.6f} E")
    print(f"  PA:      FL{int(round(geom_to_pa_ft(H_SEP)/100))} "
          f"({geom_to_pa_ft(H_SEP):.0f} ft, {isa_pressure_hPa(H_SEP):.1f} hPa)")
    print(f"  k_vert = {K_VERT:.5f} m²/kg  ->  Cd = {K_VERT*2*M_GON/A_SPH:.3f}")
    print(f"  Descent fit RMS: {rms_scan[best_i]:.1f} m")

    return dict(
        TS_OPT=TS_OPT, K_VERT=K_VERT,
        H_SEP=H_SEP, LAT_SEP=LAT_SEP, LON_SEP=LON_SEP, VH0_SEP=VH0_SEP,
        hh=hh, mm=mm, ss=ss,
        p_h=p_h, p_lat=p_lat, p_lon=p_lon, dp_h=dp_h,
        tsep_scan=tsep_scan, rms_scan=rms_scan,
        desc_fit_rms=rms_scan[best_i],
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3  PARSE IGRA2 RADIOSONDE
# ═══════════════════════════════════════════════════════════════════════════

def load_sounding(zip_path, date_str, hour):
    with zipfile.ZipFile(zip_path) as z:
        fname = [n for n in z.namelist() if n.endswith('.txt')][0]
        lines = z.read(fname).decode('latin-1').splitlines()

    tag = f' {date_str[:4]} {date_str[5:7]} {date_str[8:10]} {hour:02d}'
    start_idx = None
    for i, l in enumerate(lines):
        if l.startswith('#EZM') and tag in l:
            start_idx = i; break
    if start_idx is None:
        raise RuntimeError(f"Sounding {date_str} {hour:02d} UTC not found.")
    print(f"Sounding found at line {start_idx}: {lines[start_idx]}")

    records = []
    j = start_idx + 1
    while j < len(lines) and not lines[j].startswith('#'):
        raw = lines[j]
        if len(raw) < 48:
            j += 1; continue
        try:
            press_raw = int(raw[8:15].strip())
            gphgt_raw = int(raw[16:21].strip())
            wdir_raw  = int(raw[43:48].strip())
            wspd_raw  = int(raw[48:52].strip())
            press_hPa = press_raw / 100.0  if press_raw != -9999 else None
            gphgt_m   = gphgt_raw          if gphgt_raw != -9999 else None
            wdir      = wdir_raw           if wdir_raw  != -9999 else None
            wspd_ms   = wspd_raw / 10.0    if wspd_raw not in (-9999, 9999) else None
            if wdir is None or wspd_ms is None:
                j += 1; continue
            h = (float(gphgt_m) if gphgt_m
                 else (isa_height_from_pressure(press_hPa) if press_hPa else None))
            if h is None:
                j += 1; continue
            wdir_rad = np.radians(wdir)
            records.append({
                'h': h, 'p_hPa': press_hPa,
                'wdir': wdir, 'wspd': wspd_ms,
                'u_E': -wspd_ms * np.sin(wdir_rad),
                'v_N': -wspd_ms * np.cos(wdir_rad),
            })
        except (ValueError, IndexError):
            pass
        j += 1

    SND = pd.DataFrame(records).sort_values('h').reset_index(drop=True)
    print(f"Parsed {len(SND)} wind levels, range {SND['h'].min():.0f}–{SND['h'].max():.0f} m")
    return SND


# ═══════════════════════════════════════════════════════════════════════════
# 4  WIND PROFILE — IN-SITU FROM ASCENT TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_ascent_wind(ASC, win=11):
    half = win // 2
    h_w, u_w, v_w = [], [], []
    asc_h   = ASC['alt[m]'].values.astype(float)
    asc_t   = ASC['t_sec'].values.astype(float)
    asc_lat = ASC['lat[deg]'].values.astype(float)
    asc_lon = ASC['lon[deg]'].values.astype(float)
    for i in range(half, len(asc_t) - half):
        sl = slice(i - half, i + half + 1)
        p_lat_f = np.polyfit(asc_t[sl], asc_lat[sl], 1)
        p_lon_f = np.polyfit(asc_t[sl], asc_lon[sl], 1)
        h_w.append(asc_h[i])
        u_w.append(p_lon_f[0] * M_PER_DEG_LON)
        v_w.append(p_lat_f[0] * M_PER_DEG_LAT)
    return np.array(h_w), np.array(u_w), np.array(v_w)


def make_wind_interpolators(h_asc_wind, u_asc_wind, v_asc_wind, SND, H_SEP):
    h_top = min(H_SEP * 1.05, h_asc_wind.max())
    mask = (h_asc_wind >= 5000) & (h_asc_wind <= h_top)
    haw = h_asc_wind[mask]; uaw = u_asc_wind[mask]; vaw = v_asc_wind[mask]
    idx_s = np.argsort(haw); haw = haw[idx_s]; uaw = uaw[idx_s]; vaw = vaw[idx_s]
    u_asc_f = interp1d(haw, uaw, kind='linear', fill_value='extrapolate')
    v_asc_f = interp1d(haw, vaw, kind='linear', fill_value='extrapolate')

    U_SEP = float(u_asc_f(H_SEP)); V_SEP = float(v_asc_f(H_SEP))

    # Wind shear diagnostic: maximum |Δwind| over 1 km below burst.
    # Sharp vertical shear can cause the H0 model to fail even without an impulse
    # (the parachute couples the gondola to rapidly-changing local wind).
    h_check = np.linspace(max(H_SEP - 1000, haw.min()), H_SEP, 40)
    u_check = u_asc_f(h_check); v_check = v_asc_f(h_check)
    dspd = np.sqrt((u_check - U_SEP) ** 2 + (v_check - V_SEP) ** 2)
    WIND_SHEAR_1KM = float(dspd.max())

    print(f"\nWind at burst (h={H_SEP:.0f} m):")
    print(f"  Ascent traj.:  u_E={U_SEP:+.2f} m/s, v_N={V_SEP:+.2f} m/s")
    print(f"  Max |Δwind| over 1 km below burst: {WIND_SHEAR_1KM:.1f} m/s", end='')
    if WIND_SHEAR_1KM > 8:
        print("  <- EXTREME SHEAR: H0 test unreliable")
    elif WIND_SHEAR_1KM > 4:
        print("  <- STRONG SHEAR: H0 test may be affected")
    else:
        print()

    if SND is not None:
        SND_sorted = SND.sort_values('h')
        u_snd_f = interp1d(SND_sorted['h'], SND_sorted['u_E'], kind='linear', fill_value='extrapolate')
        v_snd_f = interp1d(SND_sorted['h'], SND_sorted['v_N'], kind='linear', fill_value='extrapolate')
        U_SND_SEP = float(u_snd_f(H_SEP)); V_SND_SEP = float(v_snd_f(H_SEP))
        delta_u   = U_SEP - U_SND_SEP;     delta_v   = V_SEP - V_SND_SEP
        delta_wind = np.sqrt(delta_u ** 2 + delta_v ** 2)
        print(f"  Sounding UTC:  u_E={U_SND_SEP:+.2f} m/s, v_N={V_SND_SEP:+.2f} m/s")
        print(f"  D|wind| = {delta_wind:.1f} m/s")
        if delta_wind > 5:
            print(f"  WARNING: wind field changed by {delta_wind:.1f} m/s — using ascent profile.")
    else:
        U_SND_SEP = V_SND_SEP = delta_u = delta_v = delta_wind = None
        print("  No sounding provided — using ascent trajectory wind only.")

    return u_asc_f, v_asc_f, U_SEP, V_SEP, U_SND_SEP, V_SND_SEP, delta_u, delta_v, delta_wind, WIND_SHEAR_1KM


# ═══════════════════════════════════════════════════════════════════════════
# 5  HORIZONTAL HYPOTHESIS TEST
# ═══════════════════════════════════════════════════════════════════════════

def simulate_horiz(dv_N, dv_E, burst, wind_fns, dt=0.02, k_horiz=None):
    """Simulate horizontal trajectory after burst.

    k_horiz: horizontal drag coefficient [m²/kg].  Defaults to burst['K_VERT']
             (vertical value) when None.  Pass a calibrated value to decouple
             horizontal drag from the vertical fit.
    """
    H_SEP    = burst['H_SEP']
    LAT_SEP  = burst['LAT_SEP']
    LON_SEP  = burst['LON_SEP']
    VH0_SEP  = burst['VH0_SEP']
    TS_OPT   = burst['TS_OPT']
    K_VERT   = burst['K_VERT']
    K_H      = k_horiz if k_horiz is not None else K_VERT   # horizontal drag coeff
    U_SEP, V_SEP = wind_fns['U_SEP'], wind_fns['V_SEP']
    u_f, v_f     = wind_fns['u_asc_f'], wind_fns['v_asc_f']

    h  = H_SEP; vv = VH0_SEP
    u_g = U_SEP + dv_E
    v_g = V_SEP + dv_N
    t = TS_OPT; lat = LAT_SEP; lon = LON_SEP
    to = [t]; lao = [lat]; loo = [lon]; ho = [h]

    while h > 100 and t < TS_OPT + 150:
        rho = isa_density(h)
        av = 9.80665 - K_VERT * rho * vv ** 2 if vv >= 0 else 9.80665 + K_VERT * rho * vv ** 2
        vv += av * dt; h -= vv * dt
        uw = float(u_f(max(h, 500))); vw = float(v_f(max(h, 500)))
        dur = u_g - uw; dvr = v_g - vw
        vrh = np.sqrt(dur ** 2 + dvr ** 2)
        if vrh > 0.001:
            ad = K_H * rho * vrh ** 2
            u_g -= ad * (dur / vrh) * dt
            v_g -= ad * (dvr / vrh) * dt
        lat += v_g * dt / M_PER_DEG_LAT
        lon += u_g * dt / M_PER_DEG_LON
        t += dt
        to.append(t); lao.append(lat); loo.append(lon); ho.append(h)
    return np.array(to), np.array(lao), np.array(loo), np.array(ho)


def hypothesis_test(DESC, burst, wind_fns, T0, wind_shear_1km=0.0):
    # Skip packets that are before the burst (TS_OPT may be > 0 relative to apogee)
    td_all = (DESC['t_sec'].values - T0).astype(float)
    post_burst = td_all >= burst['TS_OPT']
    DESC_post = DESC[post_burst].reset_index(drop=True)

    # ── Heteroscedastic weighted hypothesis test ─────────────────────────────
    # Each GPS position measurement has two independent error sources:
    #   1. GPS noise:        σ_GPS  [constant]
    #   2. Wind-model drift: shear × v_fall × t  [grows linearly with time]
    #
    # Combined per-point uncertainty:
    #   σᵢ = sqrt(σ_GPS² + (shear · v_fall · tᵢ)²)
    #
    # Using all N_HORIZ_MAX GPS points with these weights is the maximum-
    # likelihood estimator — no hard cutoff needed.  Early points (small tᵢ)
    # naturally dominate; late points are down-weighted automatically.
    rho_sep = isa_density(burst['H_SEP'])
    v_ref   = 3.0
    tau_h   = M_GON / max(burst['K_VERT'] * rho_sep * v_ref, 1e-6)  # impulse decay time [s]

    # GPS interval and initial descent speed from first post-burst packets
    dt_vals    = []
    vfall_vals = []
    for _i in range(1, min(8, len(DESC_post))):
        _dt = float(DESC_post['t_sec'].values[_i] - DESC_post['t_sec'].values[_i - 1])
        _dh = abs(float(DESC_post['alt[m]'].values[_i] - DESC_post['alt[m]'].values[_i - 1]))
        if _dt > 0:
            dt_vals.append(_dt)
            if _dh > 0:
                vfall_vals.append(_dh / _dt)
    dt_avg  = float(np.median(dt_vals))    if dt_vals    else 4.0
    v_fall  = float(np.median(vfall_vals)) if vfall_vals else 10.0

    N_HORIZ_MAX = 12

    # ── GPS outlier detection ─────────────────────────────────────────────────
    # A GPS position outlier produces a large velocity spike in one GPS interval
    # immediately followed by a reversal in the next.  Detect by looking at
    # consecutive horizontal velocity changes: if |Δv_i| > OUTLIER_K × median
    # AND |Δv_{i+1}| is also large (the reversal), flag point i as an outlier
    # and replace its position with linear interpolation from neighbours.
    raw_n = min(N_HORIZ_MAX + 4, len(DESC_post))  # look slightly past test window
    t_raw = (DESC_post['t_sec'].values[:raw_n] - T0).astype(float)
    la_raw = DESC_post['lat[deg]'].values[:raw_n].astype(float)
    lo_raw = DESC_post['lon[deg]'].values[:raw_n].astype(float)
    h_raw  = DESC_post['alt[m]'].values[:raw_n].astype(float)

    # Compute horizontal velocity for each GPS interval
    vel_E = np.zeros(raw_n); vel_N = np.zeros(raw_n)
    for _i in range(1, raw_n):
        _dt2 = float(DESC_post['t_sec'].values[_i] - DESC_post['t_sec'].values[_i - 1])
        if _dt2 > 0:
            vel_E[_i] = (lo_raw[_i] - lo_raw[_i - 1]) / _dt2 * M_PER_DEG_LON
            vel_N[_i] = (la_raw[_i] - la_raw[_i - 1]) / _dt2 * M_PER_DEG_LAT
    # Velocity change between consecutive intervals
    dvel = np.array([np.hypot(vel_E[_i] - vel_E[_i - 1], vel_N[_i] - vel_N[_i - 1])
                     for _i in range(1, raw_n)])
    med_dvel = float(np.median(dvel[dvel > 0])) if np.any(dvel > 0) else 1.0
    OUTLIER_K = 3.5   # flag if velocity jump > 3.5× median jump
    outlier_flags = np.zeros(raw_n, dtype=bool)
    for _i in range(1, len(dvel)):   # dvel[i] = Δv entering position i+1
        if dvel[_i - 1] > OUTLIER_K * med_dvel and dvel[_i] > OUTLIER_K * med_dvel:
            # Both the spike and the reversal are large → position i is the outlier
            outlier_flags[_i] = True
    n_outliers = int(np.sum(outlier_flags))
    if n_outliers:
        print(f"  GPS outlier(s) detected and interpolated: "
              + ", ".join(f"#{_i}(t={t_raw[_i]-float(burst['TS_OPT']):.0f}s,h={h_raw[_i]:.0f}m)"
                          for _i in range(raw_n) if outlier_flags[_i]))
        for _i in np.where(outlier_flags)[0]:
            if 0 < _i < raw_n - 1:
                w = 0.5  # linear interpolation
                la_raw[_i] = (1 - w) * la_raw[_i - 1] + w * la_raw[_i + 1]
                lo_raw[_i] = (1 - w) * lo_raw[_i - 1] + w * lo_raw[_i + 1]

    # All GPS points are stored for plotting; the FIT uses only n_fit points.
    n_plot = min(N_HORIZ_MAX, raw_n)
    td_plot = t_raw[:n_plot]
    lm_plot = la_raw[:n_plot]
    om_plot = lo_raw[:n_plot]
    hm_plot = h_raw[:n_plot]

    # Per-point position uncertainty [m]: GPS noise + accumulated wind-model drift
    # Use kinematic descent fit RMS as GPS noise estimate when the fit is tight.
    # If RMS < SIGMA_H (vertical GPS spec = 12 m), the kinematic model fits well
    # and RMS reflects actual GPS position noise rather than aerodynamic model error.
    # When RMS ≥ SIGMA_H the descent fit is dominated by parachute / wind-model
    # imperfections — fall back to the GPS horizontal spec SIGMA_HOR = 5 m.
    desc_rms  = burst.get('desc_fit_rms', SIGMA_HOR)
    sigma_gps = float(desc_rms) if desc_rms < SIGMA_H else SIGMA_HOR

    shear     = wind_shear_1km / 1000.0            # [m/s per m altitude]
    shear_vel = shear * v_fall                     # [m/s per s] — wind-error accumulation rate

    # Fit range: GPS points still in the "ballistic" regime.
    # Two physical limits apply simultaneously:
    #
    # 1. Wind-model drift limit (t_max_wind):
    #    Δx(t) = (shear·v_fall / τ_h) · t³/6  →  t_max_wind = (3·σ·τ_h/(shear·v_fall))^(1/3)
    #    Beyond this, accumulated wind-profile error exceeds σ_GPS.
    #
    # 2. Impulse-persistence limit (t_max_tau = τ_h/2):
    #    The horizontal impulse decays as exp(−t/τ_h).  For long τ_h (bare gondola),
    #    the impulse signal far outlasts the wind-model window, so using only
    #    t_max_wind points wastes information.  Allow fit up to τ_h/2 —
    #    at that point the impulse has decayed to 39 % of its initial value,
    #    still strongly detectable above σ_eff.
    #
    # n_fit = largest number of points satisfying both constraints.
    if shear > 0 and v_fall > 0:
        t_max_wind = (3.0 * sigma_gps * tau_h / (shear * v_fall)) ** (1.0 / 3.0)
        # For a slow-decay system (bare gondola), the impulse signal persists well
        # beyond the wind-model uncertainty window → extend fit to τ_h/2.
        # Guard: only extend when wind model drift at τ_h/2 is below 2×σ_GPS;
        # if shear is extreme the linear drift already dominates at τ_h/2 and
        # including those points injects more wind-model noise than impulse signal.
        sigma_wind_at_half_tau = shear_vel * (tau_h / 2.0)
        if sigma_wind_at_half_tau <= 2.0 * sigma_gps:
            t_max_fit = max(t_max_wind, tau_h / 2.0)
        else:
            t_max_fit = t_max_wind
        n_fit = max(3, min(n_plot, int(t_max_fit / dt_avg)))
    else:
        t_max_wind = float('inf')
        t_max_fit  = float('inf')
        n_fit = n_plot   # no shear → use all available points

    # 3. GPS-outlier hard cap: interpolated positions are not independent data.
    #    Stop the fit window just before the first detected outlier index.
    if n_outliers:
        first_out_idx = int(np.min(np.where(outlier_flags)[0]))
        if n_fit > first_out_idx:
            n_fit = max(3, first_out_idx)

    # Fit arrays (subset used for H0/H1 optimisation)
    td_h = td_plot[:n_fit]
    lm_h = lm_plot[:n_fit]
    om_h = om_plot[:n_fit]
    hm_h = hm_plot[:n_fit]
    n_h  = n_fit

    t_since_burst = td_h - float(burst['TS_OPT'])  # [s] elapsed since burst
    sigma_eff = np.sqrt(sigma_gps**2 + (shear_vel * t_since_burst)**2)
    # Effective number of GPS-noise-equivalent independent measurements
    n_eff = float(np.sum((sigma_gps / sigma_eff)**2))

    # ── Descent GPS velocity diagnostic ─────────────────────────────────────
    u_f_asc = wind_fns['u_asc_f']; v_f_asc = wind_fns['v_asc_f']
    desc_all     = DESC.reset_index(drop=True)
    td_all2      = (desc_all['t_sec'].values - T0).astype(float)
    post2        = td_all2 >= burst['TS_OPT']
    desc_post_all = desc_all[post2].reset_index(drop=True)

    # Collect GPS-velocity pairs in the altitude range relevant to the test.
    # Limit to MAX_GPS_W pairs: enough to cover burst altitude ± several km,
    # but avoiding the O(N_descent) cost for long flights.
    MAX_GPS_W = 50
    H_SEP_val = burst['H_SEP']
    gps_vs_wind = []
    h_gps_w, u_gps_w, v_gps_w = [], [], []
    for i in range(1, len(desc_post_all)):
        if len(h_gps_w) >= MAX_GPS_W:
            break
        dt2 = float(desc_post_all['t_sec'].values[i] - desc_post_all['t_sec'].values[i-1])
        if dt2 <= 0:
            continue
        h_mid = float((desc_post_all['alt[m]'].values[i] + desc_post_all['alt[m]'].values[i-1]) / 2)
        if h_mid < H_SEP_val - 8000:  # no need to sample wind far below burst
            break
        u_gps = float((desc_post_all['lon[deg]'].values[i] - desc_post_all['lon[deg]'].values[i-1]) / dt2 * M_PER_DEG_LON)
        v_gps = float((desc_post_all['lat[deg]'].values[i] - desc_post_all['lat[deg]'].values[i-1]) / dt2 * M_PER_DEG_LAT)
        u_asc = float(u_f_asc(max(h_mid, 500))); v_asc = float(v_f_asc(max(h_mid, 500)))
        delta = np.sqrt((u_gps - u_asc) ** 2 + (v_gps - v_asc) ** 2)
        gps_vs_wind.append((h_mid, u_gps, v_gps, u_asc, v_asc, delta))
        h_gps_w.append(h_mid); u_gps_w.append(u_gps); v_gps_w.append(v_gps)

    # ── Horizontal K_VERT calibration from post-burst velocity relaxation ───────
    # After burst the gondola retains its burst-point horizontal velocity and
    # decelerates toward the local wind with time constant
    #   τ_h = M / (K_VERT_horiz · ρ · v_ref)
    # The vertical K_VERT (from the descent-rate fit) may differ from the
    # horizontal K_VERT (parachute geometry is anisotropic).
    #
    # Method: compute |v_GPS(t) − v_wind(h(t))| for the first N_VCAL GPS
    # intervals after burst (observed = descent GPS; reference = ascent wind
    # profile at the same altitude).  Fit an exponential decay to obtain
    # τ_h_horiz, then derive K_VERT_horiz.
    #
    # Quality gate: accept the calibrated value only when the fit residuals
    # are dominated by the velocity signal (not wind-model drift), i.e. when
    # the fitted v0 exceeds 2×σ_v (the velocity noise floor).
    from scipy.optimize import curve_fit as _curve_fit

    N_VCAL = 12   # velocity intervals to use for calibration
    _t_v, _d_v, _s_v = [], [], []
    for _ki in range(1, min(N_VCAL + 1, len(desc_post_all))):
        _dt_k = float(desc_post_all['t_sec'].values[_ki]
                      - desc_post_all['t_sec'].values[_ki - 1])
        if _dt_k <= 0:
            continue
        _t_mid = (float(desc_post_all['t_sec'].values[_ki])
                  + float(desc_post_all['t_sec'].values[_ki - 1])) / 2.0 \
                 - T0 - float(burst['TS_OPT'])
        if _t_mid <= 0:
            continue
        _h_m = float((desc_post_all['alt[m]'].values[_ki]
                      + desc_post_all['alt[m]'].values[_ki - 1]) / 2)
        _ug  = float((desc_post_all['lon[deg]'].values[_ki]
                      - desc_post_all['lon[deg]'].values[_ki - 1])
                     / _dt_k * M_PER_DEG_LON)
        _vg  = float((desc_post_all['lat[deg]'].values[_ki]
                      - desc_post_all['lat[deg]'].values[_ki - 1])
                     / _dt_k * M_PER_DEG_LAT)
        _uw  = float(u_f_asc(max(_h_m, 500)))
        _vw  = float(v_f_asc(max(_h_m, 500)))
        _d   = float(np.hypot(_ug - _uw, _vg - _vw))
        # Velocity uncertainty: GPS position noise translated to velocity
        # plus linear wind-model drift at this time.
        _sv  = float(np.hypot(sigma_gps / _dt_k, shear_vel * max(_t_mid, 1.0)))
        _t_v.append(_t_mid); _d_v.append(_d); _s_v.append(_sv)

    K_VERT_horiz = burst['K_VERT']   # default: same as vertical
    tau_h_horiz  = tau_h              # default: formula value
    _vcal_note   = 'default (no calibration)'

    if len(_t_v) >= 4:
        try:
            def _exp_v(t, v0, tau):
                return v0 * np.exp(-t / tau)
            _popt, _pcov = _curve_fit(
                _exp_v, _t_v, _d_v,
                p0=[float(np.max(_d_v)), tau_h],
                sigma=_s_v, absolute_sigma=True,
                bounds=([0.0, 2.0], [50.0, 600.0]),
                maxfev=3000)
            _v0_fit, _tau_fit = float(_popt[0]), float(_popt[1])
            _tau_err = float(np.sqrt(max(_pcov[1, 1], 0.0)))
            _v0_noise = float(np.median(_s_v))  # typical velocity noise floor
            # Accept when: signal visible (v0 > 2×noise) AND τ well-constrained (<60% error)
            if _v0_fit > 2.0 * _v0_noise and _tau_err / max(_tau_fit, 1) < 0.6:
                tau_h_horiz  = _tau_fit
                K_VERT_horiz = M_GON / max(tau_h_horiz * rho_sep * v_ref, 1e-9)
                _vcal_note   = (f'CALIBRATED  τ_h_horiz={tau_h_horiz:.0f}s'
                                f'  ±{_tau_err:.0f}s  v0={_v0_fit:.2f}m/s'
                                f'  K_horiz={K_VERT_horiz:.5f}m²/kg')
            else:
                _vcal_note = (f'rejected  v0={_v0_fit:.2f}m/s (noise~{_v0_noise:.2f}m/s),'
                              f' τ={_tau_fit:.0f}±{_tau_err:.0f}s — keeping formula')
        except Exception as _exc:
            _vcal_note = f'fit failed ({_exc}) — keeping formula'
    else:
        _vcal_note = 'too few velocity samples — keeping formula'

    print(f"  Horiz K calibration (descent GPS vs ascent wind profile): {_vcal_note}")
    if tau_h_horiz != tau_h:
        print(f"    τ_h formula={tau_h:.0f}s → τ_h_horiz={tau_h_horiz:.0f}s"
              f"  K_vert={burst['K_VERT']:.5f} → K_horiz={K_VERT_horiz:.5f} m²/kg")

    # Use ascent profile for the simulation (GPS velocity ≠ actual wind at
    # stratospheric altitudes — at 32 km the parachute equilibration time is
    # τ = M/(K_VERT·ρ·v_rel) ≈ 15–50 s, so GPS velocity is a time-lagged
    # inertial average, not the instantaneous local wind).
    use_gps_wind = False
    wind_fns_sim = wind_fns

    def _run_h0h1(wf):
        """Run H0 and H1 with heteroscedastic per-point uncertainty sigma_eff."""
        ts0_, la0_, lo0_, _ = simulate_horiz(0., 0., burst, wf, k_horiz=K_VERT_horiz)
        la0e = np.interp(td_h, ts0_, la0_)
        lo0e = np.interp(td_h, ts0_, lo0_)
        rl0  = (lm_h - la0e) * M_PER_DEG_LAT / sigma_eff
        re0  = (om_h - lo0e) * M_PER_DEG_LON / sigma_eff
        c0   = np.sum(rl0 ** 2 + re0 ** 2)

        def _res1(params):
            dv_n, dv_e = params
            ts_, la_, lo_, _ = simulate_horiz(dv_n, dv_e, burst, wf,
                                              k_horiz=K_VERT_horiz)
            lp = np.interp(td_h, ts_, la_); op = np.interp(td_h, ts_, lo_)
            return np.concatenate([(lm_h - lp) * M_PER_DEG_LAT / sigma_eff,
                                   (om_h - op) * M_PER_DEG_LON / sigma_eff])

        r1 = least_squares(_res1, [0., 0.], bounds=([-50, -50], [50, 50]),
                           method='trf', ftol=1e-9, xtol=1e-9)
        dv_n, dv_e = r1.x
        c1 = np.sum(r1.fun ** 2)

        ts1_, la1_, lo1_, _ = simulate_horiz(dv_n, dv_e, burst, wf,
                                             k_horiz=K_VERT_horiz)
        la1e = np.interp(td_h, ts1_, la1_)
        lo1e = np.interp(td_h, ts1_, lo1_)
        rl1_s = (lm_h - la1e) * M_PER_DEG_LAT / sigma_eff
        re1_s = (om_h - lo1e) * M_PER_DEG_LON / sigma_eff
        # Return metre-scale residuals for plotting (sigma_eff cancels back out)
        return (c0, c1, dv_n, dv_e, ts0_, la0_, lo0_, la0e, lo0e,
                rl0 * sigma_eff, re0 * sigma_eff,
                ts1_, la1_, lo1_, la1e, lo1e,
                rl1_s * sigma_eff, re1_s * sigma_eff)

    # Primary result: GPS-wind (parachute) or ascent-wind (bare gondola)
    (chi2_0h, chi2_1h, DV_N, DV_E,
     ts0, la0, lo0, la0_eval, lo0_eval, res_lat0, res_lon0,
     ts1, la1, lo1, la1_eval, lo1_eval, res_lat1, res_lon1) = _run_h0h1(wind_fns_sim)

    N_obs = 2 * n_h; dof1h_nom = N_obs - 2   # nominal dof (for chi²/dof display)

    # ── Autocorrelation of H1 residuals ──────────────────────────────────────
    # After fitting the impulse, the H1 residuals should be uncorrelated noise
    # if the model is adequate.  High lag-1 autocorrelation of H1 residuals
    # indicates a remaining systematic trend (unmodelled wind error, or spurious
    # impulse absorbing wind drift) — the GPS points are not truly independent.
    #
    # Note: H0 residuals are intentionally NOT used here.  A real impulse creates
    # strongly correlated H0 residuals (all points displaced in the same direction)
    # — that autocorrelation is the SIGNAL, not a noise artefact.  Only H1
    # residuals tell us whether the model explains the systematic part.
    #
    # Effective sample size from lag-1 autocorrelation (Bartlett/AR1 approx):
    #   n_eff_ac = N × (1 − r₁) / (1 + r₁)
    r1_normed = res_lat1 / sigma_eff       # σ-normalised H1 North residuals
    # Need at least 5 fit points for a meaningful lag-1 autocorrelation estimate
    # (2–3 pairs give unreliable r₁ and wildly incorrect n_eff_ac).
    if len(r1_normed) >= 5:
        r1_ac = float(np.corrcoef(r1_normed[:-1], r1_normed[1:])[0, 1])
        n_eff_ac = len(r1_normed) * (1 - r1_ac) / max(1 + r1_ac, 0.01)
    else:
        r1_ac = float('nan'); n_eff_ac = n_eff   # too few points — fall back to weight-based N_eff

    n_eff_final = min(n_eff, n_eff_ac)
    # Effective dof for F-test: treat N_eff independent pairs of observations
    dof_eff = max(2, int(round(2 * n_eff_final)) - 2)

    # F-test with effective dof — corrects for correlated / downweighted observations
    F_h     = ((chi2_0h - chi2_1h) / 2) / (chi2_1h / dof_eff)
    p_val_h = 1 - fdist.cdf(F_h, 2, dof_eff)

    DV_MAG = np.sqrt(DV_N ** 2 + DV_E ** 2)
    AZ     = (90 - np.degrees(np.arctan2(DV_N, DV_E))) % 360
    DP_MAG = M_GON * DV_MAG

    print(f"\n=== HORIZONTAL HYPOTHESIS TEST (fit N={n_fit}, plot N={n_plot}, N_eff={n_eff:.1f}, N_eff_ac={n_eff_ac:.1f}) ===")
    _t_wind_str = f'{t_max_wind:.1f}s' if np.isfinite(t_max_wind) else '∞'
    _t_fit_str  = f'{t_max_fit:.1f}s'  if np.isfinite(t_max_fit)  else '∞'
    print(f"  σ_GPS={sigma_gps:.1f}m (desc_rms={desc_rms:.1f}m)  τ_h={tau_h:.0f}s"
          f"  t_max_wind={_t_wind_str}  t_fit_limit={_t_fit_str}"
          f"  v_fall={v_fall:.1f}m/s  shear={wind_shear_1km:.1f}m/s/km"
          f"  σ_eff: {sigma_eff[0]:.1f}→{sigma_eff[-1]:.1f} m")
    r1_str = f'{r1_ac:+.2f}' if np.isfinite(r1_ac) else 'n/a (too few pts)'
    print(f"  H1 residual autocorr r₁={r1_str}  →  n_eff_final={n_eff_final:.1f}  dof_eff={dof_eff}")
    wind_lbl = 'descent GPS wind' if use_gps_wind else 'ascent trajectory wind'
    print(f"  Wind model: {wind_lbl}")
    if gps_vs_wind and burst['K_VERT'] > 0.05:
        print("  Descent GPS velocity vs. ascent wind profile (first few points):")
        for h_m, ug, vg, ua, va, dv in gps_vs_wind[:min(n_h, 5)]:
            flag = ' <-- MISMATCH' if dv > 4 else ''
            print(f"    h={h_m:.0f}m  GPS u={ug:+.1f} v={vg:+.1f}  asc u={ua:+.1f} v={va:+.1f}  D={dv:.1f} m/s{flag}")
    print("  Fit-point residuals  (H0 → H1)  [m / σ]:")
    for _i in range(n_h):
        _rn0 = res_lat0[_i]; _re0 = res_lon0[_i]
        _rn1 = res_lat1[_i]; _re1 = res_lon1[_i]
        _sig = sigma_eff[_i]
        print(f"    #{_i}  ΔN: {_rn0:+6.1f}m ({_rn0/_sig:+.2f}σ) → {_rn1:+6.1f}m ({_rn1/_sig:+.2f}σ)"
              f"   ΔE: {_re0:+6.1f}m ({_re0/_sig:+.2f}σ) → {_re1:+6.1f}m ({_re1/_sig:+.2f}σ)")
    print(f"H0:  chi2/dof = {chi2_0h/N_obs:.2f}  ({N_obs} obs, N_eff={n_eff:.1f})")
    print(f"H1:  chi2/dof = {chi2_1h/dof1h_nom:.2f}  ({dof1h_nom} nominal dof)")
    print(f"  Dv_N = {DV_N:+.4f} m/s,  Dv_E = {DV_E:+.4f} m/s")
    print(f"  |Dv| = {DV_MAG:.4f} m/s,  |Dp| = {DP_MAG:.4f} N·s,  az = {AZ:.1f}°")
    print(f"  F(2,{dof_eff}) = {F_h:.2f},  p = {p_val_h:.2e}  [effective dof={dof_eff}]")
    verdict = ("H0 REJECTED" if p_val_h < 0.001
               else "H0 rejected" if p_val_h < 0.05
               else "H0 cannot be rejected")
    h1_quality = ("good fit" if chi2_1h / dof1h_nom < 4 else "poor fit — model inadequate")
    print(f"-> {verdict}  (H1 quality: {h1_quality}, chi2/dof={chi2_1h/dof1h_nom:.1f})")
    if wind_shear_1km > 4:
        shear_lbl = "EXTREME" if wind_shear_1km > 8 else "strong"
        print(f"   NOTE: {shear_lbl} vertical wind shear ({wind_shear_1km:.1f} m/s/km below burst).")
    dof1h = dof1h_nom   # keep alias for return dict / plot compatibility

    return dict(
        DV_N=DV_N, DV_E=DV_E, DV_MAG=DV_MAG, AZ=AZ, DP_MAG=DP_MAG,
        chi2_0h=chi2_0h, chi2_1h=chi2_1h, F_h=F_h, p_val_h=p_val_h,
        dof1h=dof1h, n_eff=n_eff, sigma_eff=sigma_eff,
        res_lat0=res_lat0, res_lon0=res_lon0,
        res_lat1=res_lat1, res_lon1=res_lon1,
        td_h=td_h, lm_h=lm_h, om_h=om_h, hm_h=hm_h,
        td_plot=td_plot, lm_plot=lm_plot, om_plot=om_plot,
        ts0=ts0, la0=la0, lo0=lo0,
        ts1=ts1, la1=la1, lo1=lo1,
        la0_eval=la0_eval, lo0_eval=lo0_eval,
        use_gps_wind=use_gps_wind,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6  FINAL SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════

N_PRE_BURST = 5   # how many ascent points to show before burst on the map


def plot_summary(burst, wind_data, hyp, SND, h_asc_wind, u_asc_wind, v_asc_wind, ASC, output_path):
    H_SEP = burst['H_SEP']; LAT_SEP = burst['LAT_SEP']; LON_SEP = burst['LON_SEP']
    hh = burst['hh']; mm = burst['mm']; ss = burst['ss']
    K_VERT = burst['K_VERT']
    delta_u = wind_data['delta_u']; delta_wind = wind_data['delta_wind']
    wind_shear_1km = wind_data.get('wind_shear_1km', 0.0)
    DV_N = hyp['DV_N']; DV_E = hyp['DV_E']; DV_MAG = hyp['DV_MAG']
    AZ = hyp['AZ']; DP_MAG = hyp['DP_MAG']
    F_h = hyp['F_h']; p_val_h = hyp['p_val_h']; dof1h = hyp['dof1h']
    chi2_0h = hyp['chi2_0h']; chi2_1h = hyp['chi2_1h']
    use_gps_wind = hyp.get('use_gps_wind', False)
    n_eff    = hyp.get('n_eff', dof1h + 2)
    sigma_eff = hyp.get('sigma_eff', np.full(dof1h + 2, SIGMA_HOR))
    N_obs = dof1h + 2  # = 2 * n_h (adaptive)
    td_h = hyp['td_h']; lm_h = hyp['lm_h']; om_h = hyp['om_h']; hm_h = hyp['hm_h']
    ts0 = hyp['ts0']; la0 = hyp['la0']; lo0 = hyp['lo0']
    ts1 = hyp['ts1']; la1 = hyp['la1']; lo1 = hyp['lo1']
    la0_eval = hyp['la0_eval']; lo0_eval = hyp['lo0_eval']
    res_lat0 = hyp['res_lat0']; res_lon0 = hyp['res_lon0']
    res_lat1 = hyp['res_lat1']; res_lon1 = hyp['res_lon1']

    fig = plt.figure(figsize=(18, 21))
    gs  = gridspec.GridSpec(3, 3, fig, left=0.07, right=0.97, top=0.93, bottom=0.05,
                            hspace=0.44, wspace=0.30, height_ratios=[1.4, 1.0, 1.0])
    ax_map  = fig.add_subplot(gs[0, :2])
    ax_wind = fig.add_subplot(gs[0,  2])
    ax_rN   = fig.add_subplot(gs[1,  0])
    ax_rE   = fig.add_subplot(gs[1,  1])
    ax_vec  = fig.add_subplot(gs[1,  2])
    ax_chi  = fig.add_subplot(gs[2,  0])
    ax_sum  = fig.add_subplot(gs[2, 1:])

    def tm(lat, lon):
        return ((np.atleast_1d(lon) - LON_SEP) * M_PER_DEG_LON,
                (np.atleast_1d(lat) - LAT_SEP) * M_PER_DEG_LAT)

    # Pre-burst ascent points
    pre = ASC.tail(N_PRE_BURST)
    pre_x, pre_y = tm(pre['lat[deg]'].values, pre['lon[deg]'].values)
    ax_map.plot(pre_x, pre_y, color=COLORS['asc'], lw=1.5, ls='-', alpha=0.7, label='Ascent (pre-burst)')
    ax_map.scatter(pre_x, pre_y, color=COLORS['asc'], s=35, zorder=6, alpha=0.8)

    td_plot_arr = hyp.get('td_plot', td_h)
    lm_plot_arr = hyp.get('lm_plot', lm_h)
    om_plot_arr = hyp.get('om_plot', om_h)
    n_fit_pts   = len(lm_h)     # points actually used in fit
    n_plot_pts  = len(lm_plot_arr)   # all visible points (fit + context)

    msk0 = (ts0 >= td_h[0] - 0.5) & (ts0 <= td_plot_arr[-1] + 3)
    msk1 = (ts1 >= td_h[0] - 0.5) & (ts1 <= td_plot_arr[-1] + 3)
    ax_map.plot(*tm(la0[msk0], lo0[msk0]), color=COLORS['h0'], lw=2, ls='--', label='H0 trajectory')
    ax_map.plot(*tm(la1[msk1], lo1[msk1]), color=COLORS['h1'], lw=2, ls='-.', label='H1 trajectory')

    # Context points outside fit window: faint, no H0-residual arrows, no labels
    for i in range(n_fit_pts, n_plot_pts):
        xm, ym = tm(lm_plot_arr[i], om_plot_arr[i])
        ax_map.scatter(float(xm), float(ym), color='#8b949e', s=40, zorder=6,
                       edgecolors='#30363d', lw=0.5, alpha=0.45)

    # Fit points: full opacity, H0-residual arrows, annotate only the first 3.
    # Labels are placed perpendicular to the GPS trajectory so that they fan out
    # even when all GPS points lie nearly collinear with the burst point.
    lim = 420
    n_labels = min(3, n_fit_pts)
    # Trajectory direction from first to last labelled GPS point
    xfit_l = [float(tm(lm_h[j], om_h[j])[0]) for j in range(n_labels)]
    yfit_l = [float(tm(lm_h[j], om_h[j])[1]) for j in range(n_labels)]
    if n_labels >= 2:
        dx_t = xfit_l[-1] - xfit_l[0]; dy_t = yfit_l[-1] - yfit_l[0]
        tn = max(np.sqrt(dx_t**2 + dy_t**2), 1.0)
        tx_l = dx_t / tn; ty_l = dy_t / tn
    else:
        tx_l, ty_l = 1.0, 0.0
    # Perpendicular direction: pick the side pointing more "outward" from burst
    cx_l = np.mean(xfit_l); cy_l = np.mean(yfit_l)
    rc = max(np.sqrt(cx_l**2 + cy_l**2), 1.0)
    ox_l = cx_l / rc; oy_l = cy_l / rc
    perp_cw  = ( ty_l, -tx_l)   # rotate trajectory 90° CW
    perp_ccw = (-ty_l,  tx_l)   # rotate trajectory 90° CCW
    if perp_cw[0]*ox_l + perp_cw[1]*oy_l >= perp_ccw[0]*ox_l + perp_ccw[1]*oy_l:
        px_l, py_l = perp_cw
    else:
        px_l, py_l = perp_ccw
    PERP_OFF  = 170    # metres perpendicular to trajectory
    ALONG_GAP = 80     # extra along-trajectory spacing to prevent overlap
    lbl_pos = []
    for k in range(n_labels):
        along = (k - (n_labels - 1) / 2.0) * ALONG_GAP
        lbl_pos.append((
            float(xfit_l[k] + px_l * PERP_OFF + tx_l * along),
            float(yfit_l[k] + py_l * PERP_OFF + ty_l * along),
        ))

    for i in range(n_fit_pts):
        xm, ym = tm(lm_h[i], om_h[i]); xm = float(xm); ym = float(ym)
        x0, y0 = tm(la0_eval[i], lo0_eval[i]); x0 = float(x0); y0 = float(y0)
        ax_map.scatter(xm, ym, color='white', s=75, zorder=7, edgecolors='#30363d', lw=0.5)
        ax_map.annotate('', xy=(xm, ym), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.2, alpha=0.7))
        if i < 3:   # label only first 3 fit points — no crowding
            t_abs_i = burst['T0_sec'] + td_h[i]
            hh_i = int(t_abs_i // 3600); mm_i = int((t_abs_i % 3600) // 60); ss_i = int(t_abs_i % 60)
            ax_lx, ax_ly = lbl_pos[i]
            ax_map.annotate(
                f'#{i}  {hh_i:02d}:{mm_i:02d}:{ss_i:02d}\n'
                f'h={hm_h[i]:.0f}m',
                xy=(xm, ym), xytext=(ax_lx, ax_ly),
                color='white', fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color='white', lw=0.8, alpha=0.6),
                bbox=dict(facecolor='#161b22', edgecolor='#30363d', alpha=0.85, boxstyle='round,pad=0.2')
            )
    ax_map.scatter([0], [0], s=500, color=COLORS['burst'], marker='*', zorder=10, edgecolors='white')
    ax_map.annotate(
        f'BURST  {hh:02d}:{mm:02d}:{ss:02d} UTC\n'
        f'{LAT_SEP:.5f}°N, {LON_SEP:.5f}°E\n'
        f'h = {H_SEP:.0f} m  (FL{int(round(geom_to_pa_ft(H_SEP)/100))})',
        xy=(0, 0), xytext=(60, 290),
        color=COLORS['burst'], fontsize=8.5, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['burst'], lw=1.2),
        bbox=dict(facecolor='#161b22', edgecolor=COLORS['burst'], alpha=0.92, boxstyle='round,pad=0.3')
    )
    ax_map.set_xlim(-lim, lim); ax_map.set_ylim(-lim, lim)
    ax_map.plot([-lim+30, -lim+130], [-lim+35, -lim+35], color='white', lw=3)
    ax_map.text(-lim+80, -lim+58, '100 m', ha='center', color='white', fontsize=8)
    ax_map.text(lim*0.87, lim*0.92, 'N↑', color='white', fontsize=13, fontweight='bold')
    ax_map.set_xlabel('E–W offset from burst [m]'); ax_map.set_ylabel('N–S offset from burst [m]')
    ax_map.set_title('Gondola positions')
    ax_map.legend(fontsize=8.5, loc='lower right')

    h_lo = max(5000, H_SEP - 7000)   # 7 km below burst, floor at 5 km
    h_hi = H_SEP + 2000               # 2 km above burst
    hasc = (h_asc_wind >= h_lo) & (h_asc_wind <= h_hi)
    ax_wind.plot(u_asc_wind[hasc], h_asc_wind[hasc] / 1000,
                 color=COLORS['asc'], lw=2, label='Ascent u_E')
    ax_wind.plot(v_asc_wind[hasc], h_asc_wind[hasc] / 1000,
                 color=COLORS['asc'], lw=1.5, ls='--', alpha=0.6, label='Ascent v_N')
    all_wind_vals = list(u_asc_wind[hasc]) + list(v_asc_wind[hasc])
    if SND is not None:
        hband = SND['h'].between(h_lo, h_hi)
        snd_hour_lbl = wind_data.get('sounding_hour', '??')
        ax_wind.plot(SND.loc[hband, 'u_E'], SND.loc[hband, 'h'] / 1000,
                     color=COLORS['snd'], lw=2, label=f'Sounding u_E ({snd_hour_lbl:02d} UTC)')
        ax_wind.plot(SND.loc[hband, 'v_N'], SND.loc[hband, 'h'] / 1000,
                     color=COLORS['snd'], lw=1.5, ls='--', alpha=0.6, label='Sounding v_N')
        all_wind_vals += list(SND.loc[hband, 'u_E']) + list(SND.loc[hband, 'v_N'])
        if delta_wind is not None:
            ax_wind.text(0.03, 0.97, f'Du_E~+{delta_u:.0f} m/s in 2h!',
                         transform=ax_wind.transAxes, color=COLORS['snd'], fontsize=8.5,
                         va='top', bbox=dict(facecolor='#161b22', edgecolor=COLORS['snd'],
                                             alpha=0.9, boxstyle='round,pad=0.3'))
    ax_wind.axhline(H_SEP / 1000, color=COLORS['burst'], lw=1.5, ls=':')
    ax_wind.axvline(0, color='#30363d', lw=0.8)
    ax_wind.set_xlabel('Wind component [m/s]'); ax_wind.set_ylabel('Altitude [km]')
    ax_wind.set_title(f'Wind profiles {h_lo/1000:.0f}–{h_hi/1000:.0f} km')
    if all_wind_vals:
        w_abs = max(abs(min(all_wind_vals)), abs(max(all_wind_vals)), 5)
        ax_wind.set_xlim(-w_abs * 1.25, w_abs * 1.25)
    ax_wind.set_ylim(h_lo / 1000, h_hi / 1000)
    ax_wind.legend(fontsize=7.5, loc='upper right', ncol=1)

    # Normalized residuals: r / σ_eff  (dimensionless).
    # ±1 band is constant — immediately shows whether model is adequate.
    # H1 good fit → points near 0, within ±1.  H1 poor fit → points at ±2 or more.
    for ax2, r0_m, r1_m, lbl in [(ax_rN, res_lat0, res_lat1, 'N'),
                                  (ax_rE, res_lon0, res_lon1, 'E')]:
        r0 = r0_m / sigma_eff   # dimensionless (σ units)
        r1 = r1_m / sigma_eff
        ax2.axhline(0, color='white', lw=0.8)
        ax2.axhspan(-1, 1, alpha=0.12, color='white', label='±1 σ_eff')
        ax2.axhline( 2, color='#ff6600', lw=0.7, ls=':', alpha=0.6)
        ax2.axhline(-2, color='#ff6600', lw=0.7, ls=':', alpha=0.6)
        dv_comp = DV_N if lbl == 'N' else DV_E
        rms0 = np.sqrt(np.mean(r0**2))
        rms1 = np.sqrt(np.mean(r1**2))
        # When the impulse component is negligible the two curves are identical —
        # skip drawing H1 and add a note instead so the plot is not misleading.
        no_impulse = abs(dv_comp) < 0.05
        ax2.plot(td_h, r0, 'o-', color=COLORS['h0'], ms=9, lw=2.2,
                 label=f'H0  RMS={rms0:.2f} σ')
        if no_impulse:
            ax2.text(0.50, 0.92, f'Dv_{lbl} = {dv_comp:.3f} m/s  →  H1 ≡ H0',
                     transform=ax2.transAxes, ha='center', va='top',
                     color='#8b949e', fontsize=8, style='italic')
        else:
            ax2.plot(td_h, r1, 's--', color=COLORS['h1'], ms=9, lw=2.2,
                     label=f'H1  RMS={rms1:.2f} σ')
        ax2.set_xlabel('t [s] from apogee')
        ax2.set_ylabel(f'Δ{lbl} / σ_eff  [dimensionless]')
        ax2.set_title(f'Normalized residuals Δ{lbl}  '
                      f'(σ_eff {sigma_eff[0]:.0f}→{sigma_eff[-1]:.0f} m, N_eff={n_eff:.1f})')
        ax2.legend(fontsize=8.5)

    ax_vec.set_aspect('equal')
    sc2 = 5
    sigma_v = np.sqrt(2) * SIGMA_HOR / 4.0
    circ = Circle((0, 0), sigma_v * sc2, fill=True, color='#30363d', alpha=0.5, zorder=1)
    ax_vec.add_patch(circ)
    ax_vec.text(0, sigma_v*sc2+1.5, f'sigma_v~{sigma_v:.1f}m/s', ha='center', va='bottom',
                color='#8b949e', fontsize=8)
    ax_vec.annotate('', xy=(DV_E * sc2, DV_N * sc2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->,head_width=0.6', color=COLORS['h1'], lw=3))
    ax_vec.annotate(
        f'|Dv|={DV_MAG:.2f} m/s\np={p_val_h:.1e}',
        xy=(DV_E * sc2, DV_N * sc2), xytext=(DV_E * sc2 + 5, DV_N * sc2 - 2),
        color=COLORS['h1'], fontsize=8, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['h1'], lw=0.8, alpha=0.7),
        bbox=dict(facecolor='#161b22', edgecolor=COLORS['h1'], alpha=0.85, boxstyle='round,pad=0.3')
    )
    lv = 22; ax_vec.set_xlim(-lv, lv); ax_vec.set_ylim(-lv, lv)
    ax_vec.plot([-lv, lv], [0, 0], color='#30363d', lw=0.5)
    ax_vec.plot([0, 0], [-lv, lv], color='#30363d', lw=0.5)
    ax_vec.text(lv - 0.5, 0.5, 'E', color='white', fontsize=10, ha='right')
    ax_vec.text(0.5, lv - 0.5, 'N', color='white', fontsize=10, fontweight='bold')
    ax_vec.set_xlabel('Dv_E [m/s × scale]'); ax_vec.set_ylabel('Dv_N [m/s × scale]')
    wind_src = 'descent GPS wind' if use_gps_wind else 'ascent wind'
    ax_vec.set_title(f'Impulse vector  [{wind_src}]')

    v0 = chi2_0h / N_obs; v1 = chi2_1h / dof1h
    bars = ax_chi.bar(['H0\n(no impulse)', 'H1\n(with impulse)'], [v0, v1],
                      color=[COLORS['h0'], COLORS['h1']], alpha=0.85, edgecolor='white', lw=0.8)
    ax_chi.axhline(1, color='white', lw=1.5, ls='--', label='Ideal fit (=1)')
    ax_chi.axhline(3, color='#ff6600', lw=1.0, ls=':', alpha=0.6, label='Poor fit (=3)')
    for bar, v in zip(bars, [v0, v1]):
        ax_chi.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f'{v:.2f}', ha='center', color='white', fontsize=12, fontweight='bold')
    ax_chi.set_ylabel('chi²/dof')
    wind_src = 'descent GPS wind' if use_gps_wind else 'ascent wind'
    ax_chi.set_title(f'Reduced chi²  [{wind_src}]')
    ax_chi.legend(fontsize=8.5)

    ax_sum.set_facecolor('#161b22'); ax_sum.axis('off')
    snd_hour_lbl = wind_data.get('sounding_hour')
    rows = [
        ('SUMMARY — HAB Horizontal impulse at gondola burst', 'white', 11, True),
        (f'Gondola: m={M_GON*1000:.0f}g, d={D_SPH*1000:.0f}mm, Cd~{K_VERT*2*M_GON/A_SPH:.2f}  ·  '
         f'Burst: {hh:02d}:{mm:02d}:{ss:02d} UTC, h={H_SEP:.0f}m, FL{int(round(geom_to_pa_ft(H_SEP)/100))}',
         '#8b949e', 8, False),
        ('', 'white', 5, False),
    ]
    wind_src_lbl = 'descent GPS velocity' if use_gps_wind else 'ascent trajectory (in-situ)'
    if delta_wind is not None:
        rows += [
            (f'WARNING  Sounding {snd_hour_lbl:02d} UTC vs ascent trajectory: D|wind| = {delta_wind:.0f} m/s at burst altitude!',
             COLORS['snd'], 9, True),
            ('   -> Sounding unsuitable as reference wind; ascent profile used instead.',
             COLORS['snd'], 8.5, False),
            ('', 'white', 5, False),
        ]
    else:
        rows += [
            (f'Wind reference: {wind_src_lbl}', '#8b949e', 8.5, False),
            ('', 'white', 5, False),
        ]
    h1_fit_lbl = (f'chi2/dof = {v1:.2f}  (good fit)' if v1 < 4
                  else f'chi2/dof = {v1:.2f}  (poor fit — model inadequate)')
    verdict_str = ('H0 REJECTED' if p_val_h < 0.001 else
                   'H0 rejected' if p_val_h < 0.05 else 'H0 cannot be rejected')
    rows += [
        (f'Horizontal test  [wind = {wind_src_lbl}]:', 'white', 9, True),
        (f'  H0: chi2/dof = {v0:.2f}', COLORS['h0'], 8.5, False),
        (f'  H1: {h1_fit_lbl}', COLORS['h1'], 8.5, False),
        (f'  F-test: F(2,{dof1h}) = {F_h:.1f},  p = {p_val_h:.2e}  ->  {verdict_str}', 'white', 9, True),
        ('', 'white', 5, False),
        ('Impulse vector:', 'white', 9, True),
        (f'  Dv = ({DV_E:+.2f} E,  {DV_N:+.2f} N) m/s   |   |Dv| = {DV_MAG:.2f} m/s   |   az = {AZ:.0f} deg',
         'white', 9, True),
        (f'  |Dp| = {DP_MAG:.2f} N·s  on {M_GON*1000:.0f} g gondola', 'white', 8.5, False),
    ]
    if wind_shear_1km > 4:
        shear_col = '#ff6600' if wind_shear_1km > 8 else '#ffa040'
        rows += [
            ('', 'white', 5, False),
            (f'Wind shear note: {wind_shear_1km:.1f} m/s over 1 km below burst altitude',
             shear_col, 8.5, False),
        ]
    y = 0.98
    for txt, col, fs, bold in rows:
        ax_sum.text(0.02, y, txt, transform=ax_sum.transAxes,
                    color=col, fontsize=fs, fontweight='bold' if bold else 'normal', va='top')
        y -= 0.057 if fs >= 9.5 else 0.044

    fig.suptitle('HAB — Final Analysis: Horizontal Impulse at Gondola Separation',
                 fontsize=13, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args():
    ap = argparse.ArgumentParser(description='HAB collision analysis')
    ap.add_argument('--flight',   default=FLIGHT_XLSX,
                    help='Flight data file (.xlsx or .txt TTS11 log)')
    ap.add_argument('--date',     default=SOUNDING_DATE,
                    help='Flight date YYYY-MM-DD (needed for sounding lookup)')
    ap.add_argument('--sounding', default=None,
                    help='IGRA2 sounding ZIP file (optional)')
    ap.add_argument('--output',   default=OUTPUT_PNG,
                    help='Output PNG path')
    return ap.parse_args()


def _nearest_sounding_hour(t_sec):
    """Return 0 or 12 — whichever standard sounding is closer to the flight time."""
    h = (t_sec % 86400) / 3600
    return 0 if min(h, 24 - h) < abs(h - 12) else 12


def main():
    args = _parse_args()
    # Normalise date to 'YYYY MM DD' regardless of separator used
    date_str = args.date.replace('-', ' ').replace('/', ' ')

    # 1 — load flight
    df = load_flight(args.flight)
    MAX_IDX = int(df['alt[m]'].idxmax())
    T0      = df.loc[MAX_IDX, 't_sec']
    ASC     = df.loc[:MAX_IDX].copy().reset_index(drop=True)
    DESC    = df.loc[MAX_IDX:].copy().reset_index(drop=True)
    print(f"Total packets: {len(df)}")
    print(f"Apogee: {df.loc[MAX_IDX,'time[UTC]']}, alt = {df.loc[MAX_IDX,'alt[m]']:.0f} m")

    # 2 — burst reconstruction
    burst = reconstruct_burst(ASC, DESC, T0)
    burst['T0_sec'] = T0

    # 3 — radiosonde (optional)
    SND = None
    sounding_hour = None
    if args.sounding:
        sounding_hour = _nearest_sounding_hour(T0)
        SND = load_sounding(args.sounding, date_str, sounding_hour)

    # 4 — wind profiles
    h_asc_wind, u_asc_wind, v_asc_wind = build_ascent_wind(ASC)
    (u_asc_f, v_asc_f,
     U_SEP, V_SEP,
     U_SND_SEP, V_SND_SEP,
     delta_u, delta_v, delta_wind,
     WIND_SHEAR_1KM) = make_wind_interpolators(
        h_asc_wind, u_asc_wind, v_asc_wind, SND, burst['H_SEP'])

    wind_fns = dict(
        u_asc_f=u_asc_f, v_asc_f=v_asc_f,
        U_SEP=U_SEP, V_SEP=V_SEP,
    )
    wind_data = dict(
        U_SND_SEP=U_SND_SEP, V_SND_SEP=V_SND_SEP,
        delta_u=delta_u, delta_v=delta_v, delta_wind=delta_wind,
        sounding_hour=sounding_hour,
        wind_shear_1km=WIND_SHEAR_1KM,
    )

    # 5 — hypothesis test
    hyp = hypothesis_test(DESC, burst, wind_fns, T0, wind_shear_1km=WIND_SHEAR_1KM)

    # 6 — summary figure
    plot_summary(burst, wind_data, hyp, SND, h_asc_wind, u_asc_wind, v_asc_wind, ASC, args.output)


if __name__ == '__main__':
    main()
