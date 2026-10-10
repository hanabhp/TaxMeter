"""taxmeter.core — constellation, geometry, link budget, workload generation.

Rebuilt from the written specification of implementation A of "The Predictability Tax".
Every numeric choice the paper does NOT pin down is a rebuild choice and is
logged in SPEC_GAPS.md at the repo root.
"""
import numpy as np
try:
    from sgp4.api import Satrec, SatrecArray, WGS72, jday
except ImportError:            # measurement-only hosts (e.g. a Raspberry Pi) do not need SGP4
    Satrec = SatrecArray = WGS72 = jday = None

# ----------------------------- constants ------------------------------------
MU_KM3_S2 = 398600.4418
RE_WGS84_KM = 6378.137
E2_WGS84 = 6.69437999014e-3
RE_SPHERE_KM = 6371.0

# Link budget (SPEC GAP: paper gives no link budget. Constants chosen to give
# RSRP ~ -69 dBm at zenith for the 550 km shell, Ku-band downlink.)
F_MHZ = 11700.0
EIRP_DBM = 66.7
G_RX_DBI = 33.0
RSRP_C = EIRP_DBM + G_RX_DBI - 32.44 - 20.0 * np.log10(F_MHZ)
SHADOW_SIGMA_DB = 2.0        # slow shadowing, 30 s coherence blocks
SHADOW_BLOCK_S = 30.0
FAST_SIGMA_DB = 3.0          # fast scintillation, iid per (city, sat, time)
MIN_ELEV_DEG = 25.0          # serving/candidate visibility threshold
GT_MIN_ELEV_DEG = 5.0        # candidate must remain above this at t+5 s
HYS_DB = 3.0                 # A3 hysteresis used for serving continuity

# Paper text: "72 orbital planes at 53.0 deg, 22 satellites per plane" = 1,584
# satellites, but the paper claims 4,408 (the full Gen-1 license across 5
# shells). Both presets provided; discrepancy logged in SPEC_GAPS.md.
SHELLS_SHELL1 = [(72, 22, 53.0, 550.0)]
SHELLS_GEN1_FULL = [
    (72, 22, 53.0, 550.0),
    (72, 22, 53.2, 540.0),
    (36, 20, 70.0, 570.0),
    (6, 58, 97.6, 560.0),
    (4, 43, 97.6, 560.0),
]

CITIES = {
    "san_francisco": (37.7749, -122.4194),
    "new_york":      (40.7128, -74.0060),
    "london":        (51.5074, -0.1278),
    "tokyo":         (35.6762, 139.6503),
    "mumbai":        (19.0760, 72.8777),
}
# twenty cities spanning 34 S to 64 N, used for the scale run
CITIES_20 = dict(CITIES, **{
    "reykjavik":    (64.1466, -21.9426), "anchorage":  (61.2181, -149.9003),
    "oslo":         (59.9139, 10.7522),  "toronto":    (43.6532, -79.3832),
    "seoul":        (37.5665, 126.9780), "mexico_city": (19.4326, -99.1332),
    "delhi":        (28.6139, 77.2090),  "lagos":      (6.5244, 3.3792),
    "singapore":    (1.3521, 103.8198),  "nairobi":    (-1.2921, 36.8219),
    "jakarta":      (-6.2088, 106.8456), "sao_paulo":  (-23.5505, -46.6333),
    "sydney":       (-33.8688, 151.2093), "cape_town":  (-33.9249, 18.4241),
    "buenos_aires": (-34.6037, -58.3816),
})

# ----------------------------- constellation --------------------------------

def build_constellation(shells, jd_epoch, fr_epoch, walker_f=1):
    """Walker-delta shells as SGP4 Satrec objects (near-circular, bstar=0)."""
    sats = []
    epoch1949 = jd_epoch + fr_epoch - 2433281.5
    satnum = 1
    for planes, per_plane, inc_deg, alt_km in shells:
        a = RE_SPHERE_KM + alt_km
        no_kozai = np.sqrt(MU_KM3_S2 / a ** 3) * 60.0  # rad/min
        n_total = planes * per_plane
        for p in range(planes):
            raan = 2.0 * np.pi * p / planes
            for s in range(per_plane):
                m = 2.0 * np.pi * ((s / per_plane) + walker_f * p / n_total)
                sat = Satrec()
                sat.sgp4init(
                    WGS72, 'i', satnum, epoch1949,
                    0.0, 0.0, 0.0,            # bstar, ndot, nddot
                    1.0e-4, 0.0,              # ecco, argpo
                    np.radians(inc_deg),      # inclo
                    m % (2.0 * np.pi),        # mo
                    no_kozai,                 # no_kozai
                    raan,                     # nodeo
                )
                sats.append(sat)
                satnum += 1
    return sats

def load_tle_constellation(path):
    """Satrec objects from a CelesTrak-style TLE file (live constellation)."""
    lines = [l.rstrip() for l in open(path) if l.strip()]
    sats = []
    for i in range(0, len(lines) - 2, 3):
        l1, l2 = lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            sats.append(Satrec.twoline2rv(l1, l2))
    return sats

# ------------------------------- geometry -----------------------------------

def gmst_rad(jd_ut1):
    d = jd_ut1 - 2451545.0
    return np.radians((280.46061837 + 360.98564736629 * d) % 360.0)

def teme_to_ecef(r_teme, gmst):
    """r_teme: (ns, nt, 3) km; gmst: (nt,) rad -> ECEF, in place-ish."""
    cg, sg = np.cos(gmst), np.sin(gmst)
    x = r_teme[..., 0]; y = r_teme[..., 1]
    xe = x * cg + y * sg
    ye = -x * sg + y * cg
    out = np.empty_like(r_teme, dtype=np.float32)
    out[..., 0] = xe; out[..., 1] = ye; out[..., 2] = r_teme[..., 2]
    return out

def site_ecef(lat_deg, lon_deg, h_km=0.0):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = RE_WGS84_KM / np.sqrt(1.0 - E2_WGS84 * np.sin(lat) ** 2)
    x = (n + h_km) * np.cos(lat) * np.cos(lon)
    y = (n + h_km) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2_WGS84) + h_km) * np.sin(lat)
    site = np.array([x, y, z], dtype=np.float64)
    up = np.array([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)], dtype=np.float64)
    return site, up

def elev_range(r_ecef, site, up):
    """-> el_deg (ns, nt) f32, rng_km (ns, nt) f32."""
    d = r_ecef - site.astype(np.float32)
    rng = np.sqrt((d * d).sum(axis=-1))
    u = (d @ up.astype(np.float32)) / np.maximum(rng, 1e-6)
    el = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))
    return el.astype(np.float32), rng.astype(np.float32)

def slant_range_km(el_deg, alt_km=550.0):
    """Spherical-Earth slant range from elevation (used by predictors)."""
    el = np.radians(np.maximum(el_deg, 0.5))
    re, r = RE_SPHERE_KM, RE_SPHERE_KM + alt_km
    return np.sqrt(r * r - (re * np.cos(el)) ** 2) - re * np.sin(el)

def rsrp_deterministic(el_deg, rng_km):
    """Deterministic RSRP component (dBm) shared by workload and predictors."""
    el = np.maximum(el_deg, 3.0)
    atm = 0.5 / np.maximum(np.sin(np.radians(el)), 0.05)
    return RSRP_C - 20.0 * np.log10(np.maximum(rng_km, 1.0)) - atm

# ------------------------------- workload -----------------------------------

def generate_workload(shells=SHELLS_SHELL1, hours=4.0, tick_s=10, hist_len=8,
                      seed=42, out="data/workload.npz", verbose=True,
                      tle_path=None, tick_offset=0, epoch=(2026, 9, 29),
                      cities=None):
    """tick_s/tick_offset: decision cadence. Starlink reconfigures at the 12th,
    27th, 42nd and 57th second of each minute, i.e. tick_s=15, tick_offset=12.
    tle_path: use a live TLE set instead of the Walker-delta shells."""
    jd0, fr0 = jday(*epoch, 0, 0, 0.0)
    if tle_path:
        sats = load_tle_constellation(tle_path)
    else:
        sats = build_constellation(shells, jd0, fr0)
    ns = len(sats)

    ticks = np.arange(tick_offset, int(hours * 3600) + 1, tick_s, dtype=np.int64)
    times = np.unique(np.concatenate([ticks, ticks + 1, ticks + 5]))
    jd = np.full(times.shape, jd0, dtype=np.float64)
    fr = fr0 + times.astype(np.float64) / 86400.0

    if verbose:
        print(f"[workload] {ns} sats, {len(times)} epochs, propagating (SGP4)...")
    arr = SatrecArray(sats)
    r_ecef = np.empty((ns, len(times), 3), dtype=np.float32)
    bad = 0
    chunk = max(1, int(6.0e7 // max(ns, 1)))       # ~60M state vectors per chunk
    for a in range(0, len(times), chunk):
        b = min(len(times), a + chunk)
        e, r, _v = arr.sgp4(jd[a:b], fr[a:b])      # r: (ns, nt, 3) km TEME
        bad += int((e != 0).sum())
        r_ecef[:, a:b, :] = teme_to_ecef(r, gmst_rad(jd[a:b] + fr[a:b]))
        del r, _v, e
    if bad:
        print(f"[workload] WARNING: {bad} propagation error flags (ignored)")

    tick_idx = np.searchsorted(times, ticks)
    idx_p1 = np.searchsorted(times, ticks + 1)
    idx_p5 = np.searchsorted(times, ticks + 5)
    blk = (times // int(SHADOW_BLOCK_S)).astype(np.int64)
    nblk = int(blk.max()) + 1

    master = np.random.default_rng(seed)
    shadow_tbl = (SHADOW_SIGMA_DB *
                  master.standard_normal((ns, nblk))).astype(np.float32)

    # accumulators
    F = {k: [] for k in ["s_rh", "s_eh", "s_el", "s_rt", "s_rg",
                          "c_rh", "c_eh", "c_el", "c_rt", "c_rg",
                          "c_mask", "label", "city", "tick", "n_vis"]}
    cities = cities or CITIES
    city_counts = {}

    for ci, (city, (lat, lon)) in enumerate(cities.items()):
        site, up = site_ecef(lat, lon)
        el, rng = elev_range(r_ecef, site, up)
        rsrp = rsrp_deterministic(el, rng) + shadow_tbl[:, blk]
        fast = (FAST_SIGMA_DB *
                np.random.default_rng(seed + 1000 + ci)
                .standard_normal((ns, len(times)))).astype(np.float32)
        rsrp = np.where(el > 0.5, rsrp + fast, np.float32(-999.0))
        del fast

        serving, n_ev = -1, 0
        sat_ids = np.arange(ns)
        for k in range(len(ticks)):
            ti, ti5 = int(tick_idx[k]), int(idx_p5[k])
            vis = el[:, ti] >= MIN_ELEV_DEG
            if not vis.any():
                serving = -1
                continue
            cur = np.where(vis, rsrp[:, ti], -999.0)
            if serving < 0 or el[serving, ti] < MIN_ELEV_DEG:
                serving = int(np.argmax(cur))

            pool = np.where(vis & (sat_ids != serving), cur, -999.0)
            order = np.argsort(pool)[::-1][:5]
            order = order[pool[order] > -500.0]
            K = len(order)
            if K >= 2:
                fut = rsrp[order, ti5].copy()
                fut[el[order, ti5] < GT_MIN_ELEV_DEG] = -999.0
                if fut.max() > -500.0:
                    label = int(np.argmax(fut))
                    hk = tick_idx[max(0, k - hist_len + 1):k + 1]
                    if len(hk) < hist_len:
                        hk = np.concatenate(
                            [np.full(hist_len - len(hk), hk[0]), hk])
                    ti1 = int(idx_p1[k])

                    def hist(mat, sid):
                        h = mat[sid, hk].astype(np.float32).copy()
                        h[h < -500.0] = -140.0
                        return h

                    c_rh = np.full((5, hist_len), -140.0, np.float32)
                    c_eh = np.zeros((5, hist_len), np.float32)
                    c_el = np.zeros(5, np.float32)
                    c_rt = np.zeros(5, np.float32)
                    c_rg = np.full(5, 2000.0, np.float32)
                    c_mk = np.zeros(5, bool)
                    for j, sid in enumerate(order):
                        c_rh[j] = hist(rsrp, sid)
                        c_eh[j] = el[sid, hk]
                        c_el[j] = el[sid, ti]
                        c_rt[j] = el[sid, ti1] - el[sid, ti]
                        c_rg[j] = rng[sid, ti]
                        c_mk[j] = True

                    F["s_rh"].append(hist(rsrp, serving))
                    F["s_eh"].append(el[serving, hk])
                    F["s_el"].append(el[serving, ti])
                    F["s_rt"].append(el[serving, ti1] - el[serving, ti])
                    F["s_rg"].append(rng[serving, ti])
                    F["c_rh"].append(c_rh); F["c_eh"].append(c_eh)
                    F["c_el"].append(c_el); F["c_rt"].append(c_rt)
                    F["c_rg"].append(c_rg); F["c_mask"].append(c_mk)
                    F["label"].append(label)
                    F["city"].append(ci); F["tick"].append(int(ticks[k]))
                    F["n_vis"].append(int(vis.sum()))
                    n_ev += 1

            # serving continuity via A3-style execution AFTER the event
            best = int(np.argmax(cur))
            if best != serving and cur[best] > cur[serving] + HYS_DB:
                serving = best

        city_counts[city] = n_ev
        if verbose:
            print(f"[workload] {city}: {n_ev} events")
        del el, rng, rsrp

    data = {k: np.asarray(v) for k, v in F.items()}
    data["label"] = data["label"].astype(np.int64)
    np.savez_compressed(out, **data)
    n = len(data["label"])
    if verbose:
        print(f"[workload] TOTAL events: {n}  (paper: 5,969)  -> {out}")
    return n, city_counts
