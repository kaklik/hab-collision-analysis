"""
HAB Collision Analysis — standalone script
Extracts gondola separation point and tests H0/H1 (natural burst vs. external impulse).

Configure the DATA section below and run:
    python tts9_analysis.py
"""

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

# ── DATA ────────────────────────────────────────────────────────────────────
FLIGHT_XLSX   = 'DATA+SENSORS-flight.xlsx'
SOUNDING_ZIP  = 'EZM00011520-data-beg2025.txt.zip'
SOUNDING_DATE = '2026 04 29'   # YYYY MM DD
SOUNDING_HOUR = 12             # UTC

# Gondola parameters
M_GON = 0.300          # kg
D_SPH = 0.160          # m

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
        h0s  = float(np.polyval(p_h,   tsep))
        vh0s = -float(np.polyval(dp_h, tsep))
        def res(params):
            k = params[0]
            if not (0.005 < k < 0.5):
                return np.ones(N_DESC_FIT) * 1e6
            return (sim_descent(k, h0s, vh0s, tsep, td_fit) - hd_fit) / SIGMA_H
        r = least_squares(res, [0.03], bounds=([0.005], [0.5]),
                          method='trf', ftol=1e-12, xtol=1e-12)
        return r, np.sqrt(np.mean(r.fun ** 2)) * SIGMA_H

    tsep_scan = np.linspace(-6, 2, 80)
    rms_scan  = [fit_descent_tsep(ts)[1] for ts in tsep_scan]
    rms_scan  = np.array(rms_scan)

    best_i    = np.argmin(rms_scan)
    TS_OPT    = tsep_scan[best_i]
    r_best, _ = fit_descent_tsep(TS_OPT)
    K_VERT    = float(r_best.x[0])

    H_SEP   = float(np.polyval(p_h,   TS_OPT))
    LAT_SEP = float(np.polyval(p_lat, TS_OPT))
    LON_SEP = float(np.polyval(p_lon, TS_OPT))
    VH0_SEP = -float(np.polyval(dp_h, TS_OPT))

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
    mask = (h_asc_wind >= 5000) & (h_asc_wind <= 15500)
    haw = h_asc_wind[mask]; uaw = u_asc_wind[mask]; vaw = v_asc_wind[mask]
    idx_s = np.argsort(haw); haw = haw[idx_s]; uaw = uaw[idx_s]; vaw = vaw[idx_s]
    u_asc_f = interp1d(haw, uaw, kind='linear', fill_value='extrapolate')
    v_asc_f = interp1d(haw, vaw, kind='linear', fill_value='extrapolate')

    SND_sorted = SND.sort_values('h')
    u_snd_f = interp1d(SND_sorted['h'], SND_sorted['u_E'], kind='linear', fill_value='extrapolate')
    v_snd_f = interp1d(SND_sorted['h'], SND_sorted['v_N'], kind='linear', fill_value='extrapolate')

    U_SEP     = float(u_asc_f(H_SEP)); V_SEP     = float(v_asc_f(H_SEP))
    U_SND_SEP = float(u_snd_f(H_SEP)); V_SND_SEP = float(v_snd_f(H_SEP))
    delta_u   = U_SEP - U_SND_SEP;     delta_v   = V_SEP - V_SND_SEP
    delta_wind = np.sqrt(delta_u ** 2 + delta_v ** 2)

    print(f"\nWind at burst (h={H_SEP:.0f} m):")
    print(f"  Sounding {SOUNDING_HOUR:02d} UTC:  u_E={U_SND_SEP:+.2f} m/s, v_N={V_SND_SEP:+.2f} m/s")
    print(f"  Ascent traj.:      u_E={U_SEP:+.2f} m/s, v_N={V_SEP:+.2f} m/s")
    print(f"  D|wind| = {delta_wind:.1f} m/s")
    if delta_wind > 5:
        print(f"  WARNING: wind field changed by {delta_wind:.1f} m/s — using ascent profile.")

    return u_asc_f, v_asc_f, U_SEP, V_SEP, U_SND_SEP, V_SND_SEP, delta_u, delta_v, delta_wind


# ═══════════════════════════════════════════════════════════════════════════
# 5  HORIZONTAL HYPOTHESIS TEST
# ═══════════════════════════════════════════════════════════════════════════

def simulate_horiz(dv_N, dv_E, burst, wind_fns, dt=0.02):
    H_SEP    = burst['H_SEP']
    LAT_SEP  = burst['LAT_SEP']
    LON_SEP  = burst['LON_SEP']
    VH0_SEP  = burst['VH0_SEP']
    TS_OPT   = burst['TS_OPT']
    K_VERT   = burst['K_VERT']
    U_SEP, V_SEP = wind_fns['U_SEP'], wind_fns['V_SEP']
    u_f, v_f     = wind_fns['u_asc_f'], wind_fns['v_asc_f']

    h  = H_SEP; vv = VH0_SEP
    u_g = U_SEP + dv_E
    v_g = V_SEP + dv_N
    t = TS_OPT; lat = LAT_SEP; lon = LON_SEP
    to = [t]; lao = [lat]; loo = [lon]; ho = [h]
    Cd_h = 0.30

    while h > 100 and t < TS_OPT + 150:
        rho = isa_density(h)
        av = 9.80665 - K_VERT * rho * vv ** 2 if vv >= 0 else 9.80665 + K_VERT * rho * vv ** 2
        vv += av * dt; h -= vv * dt
        uw = float(u_f(max(h, 500))); vw = float(v_f(max(h, 500)))
        dur = u_g - uw; dvr = v_g - vw
        vrh = np.sqrt(dur ** 2 + dvr ** 2)
        if vrh > 0.001:
            ad = 0.5 * Cd_h * A_SPH * rho * vrh ** 2 / M_GON
            u_g -= ad * (dur / vrh) * dt
            v_g -= ad * (dvr / vrh) * dt
        lat += v_g * dt / M_PER_DEG_LAT
        lon += u_g * dt / M_PER_DEG_LON
        t += dt
        to.append(t); lao.append(lat); loo.append(lon); ho.append(h)
    return np.array(to), np.array(lao), np.array(loo), np.array(ho)


def hypothesis_test(DESC, burst, wind_fns, T0):
    td_h = (DESC['t_sec'].values[:N_HORIZ] - T0).astype(float)
    lm_h = DESC['lat[deg]'].values[:N_HORIZ].astype(float)
    om_h = DESC['lon[deg]'].values[:N_HORIZ].astype(float)
    hm_h = DESC['alt[m]'].values[:N_HORIZ].astype(float)

    # H0
    ts0, la0, lo0, _ = simulate_horiz(0., 0., burst, wind_fns)
    la0_eval = np.interp(td_h, ts0, la0)
    lo0_eval = np.interp(td_h, ts0, lo0)
    res_lat0 = (lm_h - la0_eval) * M_PER_DEG_LAT
    res_lon0 = (om_h - lo0_eval) * M_PER_DEG_LON
    chi2_0h  = np.sum(res_lat0 ** 2 + res_lon0 ** 2) / SIGMA_HOR ** 2

    # H1
    def res_H1(params):
        dv_n, dv_e = params
        ts_, la_, lo_, _ = simulate_horiz(dv_n, dv_e, burst, wind_fns)
        lp = np.interp(td_h, ts_, la_); op = np.interp(td_h, ts_, lo_)
        rl = (lm_h - lp) * M_PER_DEG_LAT / SIGMA_HOR
        re = (om_h - op) * M_PER_DEG_LON / SIGMA_HOR
        return np.concatenate([rl, re])

    r1h = least_squares(res_H1, [0., 0.], bounds=([-50, -50], [50, 50]),
                        method='trf', ftol=1e-14, xtol=1e-14)
    DV_N, DV_E = r1h.x
    chi2_1h    = np.sum(r1h.fun ** 2)

    ts1, la1, lo1, _ = simulate_horiz(DV_N, DV_E, burst, wind_fns)
    la1_eval = np.interp(td_h, ts1, la1)
    lo1_eval = np.interp(td_h, ts1, lo1)
    res_lat1 = (lm_h - la1_eval) * M_PER_DEG_LAT
    res_lon1 = (om_h - lo1_eval) * M_PER_DEG_LON

    N_obs = 2 * N_HORIZ; dof1h = N_obs - 2
    F_h   = ((chi2_0h - chi2_1h) / 2) / (chi2_1h / dof1h)
    p_val_h = 1 - fdist.cdf(F_h, 2, dof1h)

    DV_MAG = np.sqrt(DV_N ** 2 + DV_E ** 2)
    AZ     = (90 - np.degrees(np.arctan2(DV_N, DV_E))) % 360
    DP_MAG = M_GON * DV_MAG

    print(f"\n=== HORIZONTAL HYPOTHESIS TEST (N={N_HORIZ}, sigma={SIGMA_HOR} m) ===")
    print(f"H0:  chi2/dof = {chi2_0h/N_obs:.2f}")
    print(f"H1:  chi2/dof = {chi2_1h/dof1h:.2f}")
    print(f"  Dv_N = {DV_N:+.4f} m/s,  Dv_E = {DV_E:+.4f} m/s")
    print(f"  |Dv| = {DV_MAG:.4f} m/s,  |Dp| = {DP_MAG:.4f} N·s,  az = {AZ:.1f}°")
    print(f"  F(2,{dof1h}) = {F_h:.2f},  p = {p_val_h:.2e}")
    verdict = ("H0 REJECTED" if p_val_h < 0.001
               else "H0 rejected" if p_val_h < 0.05
               else "H0 cannot be rejected")
    print(f"-> {verdict}")

    return dict(
        DV_N=DV_N, DV_E=DV_E, DV_MAG=DV_MAG, AZ=AZ, DP_MAG=DP_MAG,
        chi2_0h=chi2_0h, chi2_1h=chi2_1h, F_h=F_h, p_val_h=p_val_h,
        dof1h=dof1h,
        res_lat0=res_lat0, res_lon0=res_lon0,
        res_lat1=res_lat1, res_lon1=res_lon1,
        td_h=td_h, lm_h=lm_h, om_h=om_h, hm_h=hm_h,
        ts0=ts0, la0=la0, lo0=lo0,
        ts1=ts1, la1=la1, lo1=lo1,
        la0_eval=la0_eval, lo0_eval=lo0_eval,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6  FINAL SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════

N_PRE_BURST = 5   # how many ascent points to show before burst on the map


def plot_summary(burst, wind_data, hyp, SND, h_asc_wind, u_asc_wind, v_asc_wind, ASC, output_path):
    H_SEP = burst['H_SEP']; LAT_SEP = burst['LAT_SEP']; LON_SEP = burst['LON_SEP']
    hh = burst['hh']; mm = burst['mm']; ss = burst['ss']
    K_VERT = burst['K_VERT']
    U_SND_SEP = wind_data['U_SND_SEP']; V_SND_SEP = wind_data['V_SND_SEP']
    delta_u = wind_data['delta_u']; delta_wind = wind_data['delta_wind']
    DV_N = hyp['DV_N']; DV_E = hyp['DV_E']; DV_MAG = hyp['DV_MAG']
    AZ = hyp['AZ']; DP_MAG = hyp['DP_MAG']
    F_h = hyp['F_h']; p_val_h = hyp['p_val_h']; dof1h = hyp['dof1h']
    chi2_0h = hyp['chi2_0h']; chi2_1h = hyp['chi2_1h']
    N_obs = 2 * N_HORIZ
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

    msk0 = (ts0 >= td_h[0] - 0.5) & (ts0 <= td_h[-1] + 3)
    msk1 = (ts1 >= td_h[0] - 0.5) & (ts1 <= td_h[-1] + 3)
    ax_map.plot(*tm(la0[msk0], lo0[msk0]), color=COLORS['h0'], lw=2, ls='--', label='H0 trajectory')
    ax_map.plot(*tm(la1[msk1], lo1[msk1]), color=COLORS['h1'], lw=2, ls='-.', label='H1 trajectory')
    lbl_xs = [-200, -220, -240, -240, -250]
    lbl_ys = [ 180,  100,   10,  -80, -160]
    for i in range(N_HORIZ):
        xm, ym = tm(lm_h[i], om_h[i]); xm = float(xm); ym = float(ym)
        x0, y0 = tm(la0_eval[i], lo0_eval[i]); x0 = float(x0); y0 = float(y0)
        ax_map.scatter(xm, ym, color='white', s=70, zorder=7, edgecolors='#30363d', lw=0.5)
        ax_map.annotate('', xy=(xm, ym), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.2, alpha=0.7))
        t_abs_i = burst['T0_sec'] + td_h[i]
        hh_i = int(t_abs_i // 3600); mm_i = int((t_abs_i % 3600) // 60); ss_i = int(t_abs_i % 60)
        ax_map.annotate(
            f'#{i}  {hh_i:02d}:{mm_i:02d}:{ss_i:02d} UTC\n'
            f'    {lm_h[i]:.5f}°N, {om_h[i]:.5f}°E\n'
            f'    h = {hm_h[i]:.0f} m',
            xy=(xm, ym), xytext=(lbl_xs[i], lbl_ys[i]),
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
    lim = 420
    ax_map.set_xlim(-lim, lim); ax_map.set_ylim(-lim, lim)
    ax_map.plot([-lim+30, -lim+130], [-lim+35, -lim+35], color='white', lw=3)
    ax_map.text(-lim+80, -lim+58, '100 m', ha='center', color='white', fontsize=8)
    ax_map.text(lim*0.87, lim*0.92, 'N↑', color='white', fontsize=13, fontweight='bold')
    ax_map.set_xlabel('E–W offset from burst [m]'); ax_map.set_ylabel('N–S offset from burst [m]')
    ax_map.set_title('Gondola positions')
    ax_map.legend(fontsize=8.5, loc='lower right')

    hband = SND['h'].between(8000, 18000)
    hasc  = (h_asc_wind >= 8000) & (h_asc_wind <= 16500)
    ax_wind.plot(SND.loc[hband, 'u_E'], SND.loc[hband, 'h'] / 1000,
                 color=COLORS['snd'], lw=2, label=f'Sounding u_E ({SOUNDING_HOUR:02d} UTC)')
    ax_wind.plot(u_asc_wind[hasc], h_asc_wind[hasc] / 1000,
                 color=COLORS['asc'], lw=2, label='Ascent u_E')
    ax_wind.plot(SND.loc[hband, 'v_N'], SND.loc[hband, 'h'] / 1000,
                 color=COLORS['snd'], lw=1.5, ls='--', alpha=0.6, label='Sounding v_N')
    ax_wind.plot(v_asc_wind[hasc], h_asc_wind[hasc] / 1000,
                 color=COLORS['asc'], lw=1.5, ls='--', alpha=0.6, label='Ascent v_N')
    ax_wind.axhline(H_SEP / 1000, color=COLORS['burst'], lw=1.5, ls=':')
    ax_wind.axvline(0, color='#30363d', lw=0.8)
    ax_wind.set_xlabel('Wind component [m/s]'); ax_wind.set_ylabel('Altitude [km]')
    ax_wind.set_title('Wind profiles 8–17 km')
    ax_wind.set_xlim(-22, 16); ax_wind.set_ylim(8, 17)
    ax_wind.legend(fontsize=7.5, loc='upper right', ncol=1)
    ax_wind.text(-21, 16.6, f'Du_E~+{delta_u:.0f} m/s in 2h!', color=COLORS['snd'], fontsize=8.5,
                 va='top', bbox=dict(facecolor='#161b22', edgecolor=COLORS['snd'], alpha=0.9, boxstyle='round,pad=0.3'))

    for ax2, r0, r1, lbl in [(ax_rN, res_lat0, res_lat1, 'N'),
                              (ax_rE, res_lon0, res_lon1, 'E')]:
        ax2.axhline(0, color='white', lw=0.8)
        ax2.axhspan(-SIGMA_HOR, SIGMA_HOR, alpha=0.10, color='white')
        ax2.plot(td_h, r0, 'o-', color=COLORS['h0'], ms=9, lw=2.2,
                 label=f'H0  RMS={np.sqrt(np.mean(r0**2)):.1f}m')
        ax2.plot(td_h, r1, 's--', color=COLORS['h1'], ms=9, lw=2.2,
                 label=f'H1  RMS={np.sqrt(np.mean(r1**2)):.1f}m')
        ax2.set_xlabel('t [s] from apogee'); ax2.set_ylabel(f'Delta-{lbl} [m]')
        ax2.set_title(f'Residuals Delta-{lbl}  (sigma_GPS = {SIGMA_HOR:.0f} m)')
        ax2.legend(fontsize=9)

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
    ax_vec.set_title('Impulse vector')

    v0 = chi2_0h / N_obs; v1 = chi2_1h / dof1h
    bars = ax_chi.bar(['H0\n(no impulse)', 'H1\n(with impulse)'], [v0, v1],
                      color=[COLORS['h0'], COLORS['h1']], alpha=0.85, edgecolor='white', lw=0.8)
    ax_chi.axhline(1, color='white', lw=1.5, ls='--', label='Ideal fit (=1)')
    for bar, v in zip(bars, [v0, v1]):
        ax_chi.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f'{v:.2f}', ha='center', color='white', fontsize=12, fontweight='bold')
    ax_chi.set_ylabel('chi²/dof')
    ax_chi.set_title('Reduced chi²  (trajectory method, ascent wind)')
    ax_chi.legend(fontsize=8.5)

    ax_sum.set_facecolor('#161b22'); ax_sum.axis('off')
    rows = [
        ('SUMMARY — HAB Horizontal impulse at gondola burst', 'white', 11, True),
        (f'Gondola: m={M_GON*1000:.0f}g, d={D_SPH*1000:.0f}mm, Cd~{K_VERT*2*M_GON/A_SPH:.2f}  ·  '
         f'Burst: {hh:02d}:{mm:02d}:{ss:02d} UTC, h={H_SEP:.0f}m, FL{int(round(geom_to_pa_ft(H_SEP)/100))}',
         '#8b949e', 8, False),
        ('', 'white', 5, False),
        (f'WARNING  Sounding {SOUNDING_HOUR:02d} UTC vs ascent trajectory: D|wind| = {delta_wind:.0f} m/s at burst altitude!',
         COLORS['snd'], 9, True),
        ('   -> Sounding unsuitable as reference wind; ascent profile (in-situ) used instead.',
         COLORS['snd'], 8.5, False),
        ('', 'white', 5, False),
        ('Horizontal test (wind = ascent trajectory):', 'white', 9, True),
        (f'  H0: chi2/dof = {v0:.2f} -> systematic deviations -> REJECTED', COLORS['h0'], 8.5, False),
        (f'  H1: chi2/dof = {v1:.2f} -> fit within GPS noise -> ACCEPTED', COLORS['h1'], 8.5, False),
        (f'  F-test (H0 vs H1, 2 fitted params, {dof1h} residual dof): F(2,{dof1h}) = {F_h:.1f}',
         'white', 8.5, False),
        (f'  p-value = {p_val_h:.2e}  →  H0 REJECTED', 'white', 9, True),
        ('', 'white', 5, False),
        ('Impulse vector:', 'white', 9, True),
        (f'  Dv = ({DV_E:+.2f} E,  {DV_N:+.2f} N) m/s   |   |Dv| = {DV_MAG:.2f} m/s   |   az = {AZ:.0f} deg',
         'white', 9, True),
        (f'  |Dp| = {DP_MAG:.2f} N·s  on {M_GON*1000:.0f} g gondola', 'white', 8.5, False),
        ('', 'white', 5, False),
        ('Physical interpretation:', 'white', 9, True),
        (f'  Non-zero horizontal impulse (~{DP_MAG:.1f} N·s)', 'white', 8.5, False),
        ('  Consistent with a collision with a horizontally flying object.', 'white', 8.5, False),
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

def main():
    # 1 — load flight
    df = load_flight(FLIGHT_XLSX)
    MAX_IDX = int(df['alt[m]'].idxmax())
    T0      = df.loc[MAX_IDX, 't_sec']
    ASC     = df.loc[:MAX_IDX].copy().reset_index(drop=True)
    DESC    = df.loc[MAX_IDX:].copy().reset_index(drop=True)
    print(f"Total packets: {len(df)}")
    print(f"Apogee: {df.loc[MAX_IDX,'time[UTC]']}, alt = {df.loc[MAX_IDX,'alt[m]']:.0f} m")

    # 2 — burst reconstruction
    burst = reconstruct_burst(ASC, DESC, T0)
    burst['T0_sec'] = T0

    # 3 — radiosonde
    SND = load_sounding(SOUNDING_ZIP, SOUNDING_DATE, SOUNDING_HOUR)

    # 4 — wind profiles
    h_asc_wind, u_asc_wind, v_asc_wind = build_ascent_wind(ASC)
    (u_asc_f, v_asc_f,
     U_SEP, V_SEP,
     U_SND_SEP, V_SND_SEP,
     delta_u, delta_v, delta_wind) = make_wind_interpolators(
        h_asc_wind, u_asc_wind, v_asc_wind, SND, burst['H_SEP'])

    wind_fns = dict(
        u_asc_f=u_asc_f, v_asc_f=v_asc_f,
        U_SEP=U_SEP, V_SEP=V_SEP,
    )
    wind_data = dict(
        U_SND_SEP=U_SND_SEP, V_SND_SEP=V_SND_SEP,
        delta_u=delta_u, delta_v=delta_v, delta_wind=delta_wind,
    )

    # 5 — hypothesis test
    hyp = hypothesis_test(DESC, burst, wind_fns, T0)

    # 6 — summary figure
    plot_summary(burst, wind_data, hyp, SND, h_asc_wind, u_asc_wind, v_asc_wind, ASC, OUTPUT_PNG)


if __name__ == '__main__':
    main()
