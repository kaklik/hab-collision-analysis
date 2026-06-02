# Forensic Ballistic Analysis of HAB Flight

Jupyter notebook that analyses GPS telemetry from the **Dotknisevesmiru stratospheric balloon flight (29 April 2026)** to determine whether the gondola sustained a horizontal impulse consistent with a mid-air collision at burst altitude.

## Background

The balloon flight was a student balloon flight carried out by Gymnázium Matyáše Lercha Brno as part of the [Dotkni se vesmíru](https://dotknisevesmiru.cz/) project. The gondola was a polystyrene sphere with a student-built payload consisting of a Raspberry Pi, camera, UV and temperature sensors, and a borrowed [AIRDOS03](https://docs.dos.ust.cz/airdos/AIRDOS03) particle detector. AIRDOS03 is an engineering prototype that serves as the basis for SPACEDOS04, a detector planned for flight to the ISS with Czech astronaut Aleš Svoboda.

### Separation anomaly

At ~15 km altitude the rope connecting the gondola to the balloon snapped. The gondola fell without a parachute and without any balloon remnants. The payload was recovered after roughly two hours of searching. Due to a power-supply fault the AIRDOS03 stopped recording data shortly after launch; the GPS telemetry remained intact throughout the flight.

Post-flight analysis by the detector team raised the hypothesis that the balloon had been struck by an aircraft at burst altitude. This notebook is the quantitative investigation of that hypothesis.

## Flight overview

| Parameter | Value |
|-----------|-------|
| Launch | 13:21 UTC, Prague (50.008°N, 14.447°E, 316 m AMSL) |
| Burst | 14:09:28 UTC, 49.700°N, 14.594°E, 15 087 m |
| Pressure altitude | FL495 (118.8 hPa) |
| Gondola | Sphere, d = 160 mm, m = 300 g, Cd ≈ 0.31 |

## Analysis structure

1. **Data loading** — flight telemetry from `DATA+SENSORS-flight.xlsx` (809 GPS packets at 4 s intervals)
2. **Wind profile** — Prague-Libuš radiosonde (IGRA2, 12 UTC) vs. in-situ wind derived from the ascent trajectory; sounding was 10.4 m/s off at burst altitude → ascent profile used as reference
3. **Kinematic burst reconstruction** — ascent polynomial fit (degree 2, last 30 packets) intersected with kinematic free-fall model (first 25 descent packets) to locate the exact separation point in time and space
4. **Horizontal F-test** — H₀ (natural burst, wind drift only) vs. H₁ (horizontal impulse Δv fitted with `least_squares`); F-test over 5 post-burst packets
5. **Direct GPS method** — model-free velocity difference across burst as cross-validation

## Key results

| Quantity | Value |
|----------|-------|
| Burst time | 14:09:28 UTC |
| Burst altitude | 15 087 m (FL495) |
| Horizontal impulse \|Δv\| | **3.24 m/s** (trajectory fit) |
| Impulse azimuth | **191° from north (~southward)** |
| Momentum change \|Δp\| | **0.97 N·s** |
| F-test p-value | **3.5 × 10⁻⁵** → H₀ rejected |
| Direct method \|Δv\| | 2.48 m/s, az 206°, SNR 1.4 σ |

The northward component is statistically significant (t-test p = 0.038); the hypothesis of natural burst is rejected at the 0.005% level. The magnitude and direction of the impulse are consistent with a collision with a horizontally flying object.

![Summary figure](TTS9_final_analysis.png)

## Recovery photographs

| | |
|---|---|
| ![Gondola after impact](doc/img/20260429_180036.jpg) | ![Electronics laid out](doc/img/20260430_144646.jpg) |
| Gondola recovered in grass after uncontrolled free-fall (29 Apr, evening). The polystyrene sphere is cracked open. | Internal electronics laid out after opening the sphere (30 Apr). Both halves of the sphere visible; payload destroyed on impact. |

![Snapped suspension cord and flight board](doc/img/20260430_144659.jpg)

Close-up of the snapped yellow suspension cord (tangled around the attachment bolts) and the Dotkni se vesmíru MAIN v1.1 RevB flight board. The cord failure — with no parachute and no balloon remnants found — is the primary physical anomaly that motivated this analysis.

## Repository contents

```
TTS9_analysis.ipynb               # Main analysis notebook
DATA+SENSORS-flight.xlsx          # GPS telemetry (809 packets)
EZM00011520-data-beg2025.txt.zip  # IGRA2 radiosonde archive (Prague-Libuš, 2025–2026)
TTS9_final_analysis.png           # Summary figure (output)
```

## Requirements

```
python >= 3.9
numpy
pandas
matplotlib
scipy
openpyxl      # for reading .xlsx
```

Install with:

```bash
pip install numpy pandas matplotlib scipy openpyxl
```

## Running

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=120 TTS9_analysis.ipynb
```

Or open in JupyterLab / VS Code and run all cells.

## Methods

### Burst location reconstruction

The vertical position of the gondola is described by a 2nd-degree polynomial fit over the last 30 ascent packets and a kinematic free-fall model for the first 25 descent packets. The separation time `t_sep` that minimises the altitude RMS of the descent fit is found by a 1-D sweep, then refined with `scipy.optimize.least_squares`. The fitted drag coefficient Cd ≈ 0.305 is consistent with Stokes–Oseen theory for a sphere at Re ≈ 130 000.

### Horizontal hypothesis test

After burst the gondola drifts horizontally at the local wind speed while aerodynamic drag equalises its velocity with the surrounding air. Under H₁, an additional impulse (Δv_N, Δv_E) is added at `t_sep`. Both hypotheses are propagated numerically and compared against the first 5 GPS descent packets. The improvement in χ² is evaluated with an F-test (2 extra parameters, N = 10 observations).

### Wind reference

The radiosonde sounding from 12 UTC differed from the in-situ balloon-derived wind by 10.4 m/s at burst altitude. The in-situ wind profile (sliding-window linear regression over the ascent GPS positions) is used as the reference in the trajectory simulation.
