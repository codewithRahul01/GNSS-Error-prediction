"""
Stage 0 — Auto-fetch daily IRNSS/NavIC broadcast ephemeris
Tries IGS mirror servers (no auth required, not blocked by GitHub runners).
Falls back to NASA CDDIS only if mirrors fail (requires EARTHDATA_USER/PASS).

Mirror priority (fastest / most reliable first):
  1. BKG Frankfurt  — https://igs.bkg.bund.de  (no auth, open)
  2. UNAVCO Boulder — https://data.unavco.org   (no auth, open)
  3. IGN Paris      — https://igs.ign.fr        (no auth, open)
  4. NASA CDDIS     — https://cddis.nasa.gov    (Bearer token auth needed)

Usage:
    python3 mlops/stage0_fetch.py                    # fetch yesterday (safest)
    python3 mlops/stage0_fetch.py --date 2025-09-08  # specific date
    python3 mlops/stage0_fetch.py --dry-run          # show URLs, don't download
    python3 mlops/stage0_fetch.py --force            # re-download if CSV exists

Env vars (only needed if all mirrors fail and CDDIS fallback is used):
    EARTHDATA_USER   NASA Earthdata username
    EARTHDATA_PASS   NASA Earthdata password
"""

import sys, os, re, gzip, math, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger("stage0")

# ── satellite mapping ────────────────────────────────────────────────────────
SAT_MAP = {"I01": "GEO_A", "I04": "MEO_B", "I06": "MEO_C"}

# ── orbital constants ────────────────────────────────────────────────────────
MU      = 3.986005e14        # m³/s²
OMEGA_E = 7.2921151467e-5    # rad/s
C       = 2.99792458e8       # m/s
F       = -4.442807633e-10   # relativistic correction factor

DATA_DIR = ROOT / "Data" / "Raw"

# ── URL builders for each mirror ─────────────────────────────────────────────
def _filename(date: datetime) -> str:
    yr4 = date.strftime("%Y")
    doy = date.timetuple().tm_yday
    return f"BRDC00IGS_R_{yr4}{doy:03d}0000_01D_MN.rnx.gz"

def mirror_urls(date: datetime) -> list[tuple[str, str]]:
    """
    Returns list of (label, url) to try in order.
    All point to the same IGS combined broadcast ephemeris file.
    """
    yr4  = date.strftime("%Y")
    yr2  = date.strftime("%y")
    doy  = date.timetuple().tm_yday
    fname = _filename(date)

    urls = [
        # BKG (Germany) — open, no auth, reliable from GitHub runners
        ("BKG",
         f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{yr4}/{doy:03d}/{fname}"),

        # UNAVCO (USA) — open, no auth
        ("UNAVCO",
         f"https://data.unavco.org/archive/gnss/products/brdcfile/{yr4}/{doy:03d}/{fname}"),

        # IGN (France) — open, no auth
        ("IGN",
         f"https://igs.ign.fr/pub/igs/data/campaign/mgex/daily/rinex3/{yr4}/{doy:03d}/{yr2}p/{fname}"),

        # NASA CDDIS — requires Bearer token (fallback only)
        ("CDDIS",
         f"https://cddis.nasa.gov/archive/gnss/data/daily/{yr4}/{doy:03d}/{yr2}p/{fname}"),
    ]
    return urls

# ── NASA Earthdata Bearer token (CDDIS fallback only) ───────────────────────
def get_earthdata_token(user: str, pwd: str) -> str:
    token_url = "https://urs.earthdata.nasa.gov/api/users/tokens"
    resp = requests.get(token_url, auth=(user, pwd), timeout=30)
    if resp.status_code == 200:
        tokens = resp.json()
        if tokens:
            log.info("Using existing Earthdata token")
            return tokens[0]["access_token"]
    create_url = "https://urs.earthdata.nasa.gov/api/users/token"
    resp = requests.post(create_url, auth=(user, pwd), timeout=30)
    if resp.status_code in (200, 201):
        token = resp.json().get("access_token", "")
        if token:
            log.info("Created new Earthdata token")
            return token
    log.error(f"Earthdata token failed: {resp.status_code} {resp.text[:200]}")
    return ""

# ── download with mirror fallback ────────────────────────────────────────────
def download(date: datetime, dest: Path) -> bool:
    """
    Try each mirror in order. Returns True on success.
    For CDDIS uses Bearer token if credentials are available.
    """
    mirrors = mirror_urls(date)
    user = os.environ.get("EARTHDATA_USER", "")
    pwd  = os.environ.get("EARTHDATA_PASS", "")
    _token = None  # lazy-load only if CDDIS is tried

    for label, url in mirrors:
        log.info(f"Trying {label}: {url}")
        headers = {"User-Agent": "GNSS-MLOps-Pipeline/1.0 (open-source research)"}

        # CDDIS requires Bearer token
        if label == "CDDIS":
            if not user or not pwd:
                log.warning("CDDIS requires EARTHDATA_USER/PASS — skipping (not set)")
                continue
            if _token is None:
                _token = get_earthdata_token(user, pwd)
            if _token:
                headers["Authorization"] = f"Bearer {_token}"
            else:
                log.warning("Could not get Earthdata token — skipping CDDIS")
                continue

        try:
            resp = requests.get(url, headers=headers, timeout=120, allow_redirects=True)

            # detect HTML error page (auth redirect or 404 page)
            ct = resp.headers.get("Content-Type", "")
            is_html = ("text/html" in ct or
                       (len(resp.content) < 2000 and b"<!DOCTYPE" in resp.content[:100]))

            if resp.status_code == 404:
                log.warning(f"{label}: file not found (404) — trying next mirror")
                continue
            if resp.status_code != 200:
                log.warning(f"{label}: HTTP {resp.status_code} — trying next mirror")
                continue
            if is_html:
                log.warning(f"{label}: got HTML page (auth/redirect) — trying next mirror")
                continue

            # success — verify it looks like a gzip file
            if not resp.content[:2] == b'\x1f\x8b':
                log.warning(f"{label}: response doesn't look like gzip — trying next mirror")
                continue

            dest.write_bytes(resp.content)
            log.info(f"Downloaded {len(resp.content)//1024} KB from {label} → {dest.name}")
            return True

        except requests.exceptions.ConnectionError as e:
            log.warning(f"{label}: connection error ({e}) — trying next mirror")
        except requests.exceptions.Timeout:
            log.warning(f"{label}: timed out — trying next mirror")
        except requests.RequestException as e:
            log.warning(f"{label}: {e} — trying next mirror")

    log.error("All mirrors failed. Cannot fetch data.")
    log.error("Solutions:")
    log.error("  1. Fetch locally: python3 mlops/stage0_fetch.py && git push")
    log.error("  2. Set EARTHDATA_USER/PASS secrets for CDDIS fallback")
    return False

# ── RINEX 3 parser ───────────────────────────────────────────────────────────
def parse_rinex_irnss(raw_text: str, date: datetime) -> list[dict]:
    target_sats = set(SAT_MAP.keys())
    records = []
    lines = raw_text.splitlines()

    i = 0
    while i < len(lines):
        if "END OF HEADER" in lines[i]:
            i += 1
            break
        i += 1

    while i < len(lines):
        line = lines[i]
        if len(line) < 23 or line[0] != "I":
            i += 1
            continue
        prn = line[0:3].strip()
        if prn not in target_sats:
            i += 1
            continue

        try:
            yr = int(line[4:8]);  mo = int(line[9:11]);  dy = int(line[12:14])
            hr = int(line[15:17]); mn = int(line[18:20]); sc = float(line[21:23])
            ep = datetime(yr, mo, dy, hr, mn, int(sc), tzinfo=timezone.utc)
        except (ValueError, IndexError):
            i += 1
            continue

        if ep.date() != date.date():
            i += 1
            continue

        try:
            a0 = float(line[23:42].replace("D","e").replace("d","e"))
            a1 = float(line[42:61].replace("D","e").replace("d","e"))
            a2 = float(line[61:80].replace("D","e").replace("d","e"))
        except (ValueError, IndexError):
            i += 1
            continue

        orb = []
        for _ in range(7):
            i += 1
            if i >= len(lines):
                break
            ln = lines[i]
            for start in (4, 23, 42, 61):
                try:
                    orb.append(float(ln[start:start+19].replace("D","e").replace("d","e")))
                except (ValueError, IndexError):
                    orb.append(0.0)

        if len(orb) < 24:
            i += 1
            continue

        try:
            Crs=orb[1]; delta_n=orb[2]; M0=orb[3]
            Cuc=orb[4]; e=orb[5];       Cus=orb[6]; sqrtA=orb[7]
            toe=orb[8]; Cic=orb[9];     OMEGA0=orb[10]; Cis=orb[11]
            i0=orb[12]; Crc=orb[13];    omega=orb[14]; OMEGADOT=orb[15]
            IDOT=orb[16]
        except IndexError:
            i += 1
            continue

        records.append({
            "prn": prn, "epoch": ep,
            "a0": a0, "a1": a1, "a2": a2,
            "Crs": Crs, "delta_n": delta_n, "M0": M0,
            "Cuc": Cuc, "e": e, "Cus": Cus, "sqrtA": sqrtA,
            "toe": toe, "Cic": Cic, "OMEGA0": OMEGA0, "Cis": Cis,
            "i0": i0, "Crc": Crc, "omega": omega, "OMEGADOT": OMEGADOT,
            "IDOT": IDOT,
        })
        i += 1

    log.info(f"Parsed {len(records)} IRNSS records for {date.date()}")
    return records

# ── Kepler solver ────────────────────────────────────────────────────────────
def kepler_solve(M: float, e: float, tol=1e-12) -> float:
    E = M
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E

def compute_position_clock(rec: dict, t_utc: datetime) -> tuple:
    A  = rec["sqrtA"] ** 2
    n0 = math.sqrt(MU / A**3)
    n  = n0 + rec["delta_n"]
    tk = (t_utc - rec["epoch"]).total_seconds()
    M  = rec["M0"] + n * tk
    E  = kepler_solve(M, rec["e"])
    nu = math.atan2(
        math.sqrt(1 - rec["e"]**2) * math.sin(E),
        math.cos(E) - rec["e"]
    )
    phi = nu + rec["omega"]
    du  = rec["Cus"]*math.sin(2*phi) + rec["Cuc"]*math.cos(2*phi)
    dr  = rec["Crs"]*math.sin(2*phi) + rec["Crc"]*math.cos(2*phi)
    di  = rec["Cis"]*math.sin(2*phi) + rec["Cic"]*math.cos(2*phi)
    u   = phi + du
    r   = A*(1 - rec["e"]*math.cos(E)) + dr
    inc = rec["i0"] + di + rec["IDOT"]*tk
    x_orb = r*math.cos(u);  y_orb = r*math.sin(u)
    OMEGA = rec["OMEGA0"] + (rec["OMEGADOT"] - OMEGA_E)*tk - OMEGA_E*rec["toe"]
    x = x_orb*math.cos(OMEGA) - y_orb*math.cos(inc)*math.sin(OMEGA)
    y = x_orb*math.sin(OMEGA) + y_orb*math.cos(inc)*math.cos(OMEGA)
    z = y_orb*math.sin(inc)
    dt_rel = F*rec["e"]*rec["sqrtA"]*math.sin(E)
    dt_sv  = rec["a0"] + rec["a1"]*tk + rec["a2"]*tk**2 + dt_rel
    return x, y, z, dt_sv * C

def records_to_csv(records: list, date: datetime) -> pd.DataFrame:
    rows = []
    for rec in records:
        sat_id = SAT_MAP[rec["prn"]]
        x, y, z, clk = compute_position_clock(rec, rec["epoch"])
        rows.append({
            "satellite_id":      sat_id,
            "utc_time":          rec["epoch"].strftime("%Y-%m-%d %H:%M:%S"),
            "_x": x, "_y": y, "_z": z,
            "satclockerror (m)": round(clk, 6),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col, ecol in [("_x","x_error (m)"),("_y","y_error (m)"),("_z","z_error (m)")]:
        means = df.groupby("satellite_id")[col].transform("mean")
        df[ecol] = (df[col] - means).round(6)
    return df[["satellite_id","utc_time",
               "x_error (m)","y_error (m)","z_error (m)","satclockerror (m)"]]

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 0 — fetch IRNSS data")
    parser.add_argument("--date",    default=None,
        help="Date YYYY-MM-DD (default: yesterday UTC)")
    parser.add_argument("--dry-run", action="store_true",
        help="Print URLs and exit without downloading")
    parser.add_argument("--force",   action="store_true",
        help="Re-download even if CSV already exists")
    args = parser.parse_args()

    fetch_date = (datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                  if args.date
                  else datetime.now(timezone.utc) - timedelta(days=1))

    fname   = _filename(fetch_date)
    out_csv = DATA_DIR / f"irnss_{fetch_date.strftime('%Y-%m-%d')}.csv"

    log.info(f"Target date : {fetch_date.date()}")
    log.info(f"RINEX file  : {fname}")
    log.info(f"Output CSV  : {out_csv}")

    if args.dry_run:
        print("\nMirror URLs to try (in order):")
        for label, url in mirror_urls(fetch_date):
            print(f"  {label:8s} {url}")
        return

    if out_csv.exists() and not args.force:
        log.info("CSV already exists. Use --force to re-fetch.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_gz = ROOT / "mlops" / "database" / fname

    if not download(fetch_date, tmp_gz):
        sys.exit(1)

    log.info("Decompressing…")
    with gzip.open(tmp_gz, "rt", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    tmp_gz.unlink(missing_ok=True)

    records = parse_rinex_irnss(raw, fetch_date)
    if not records:
        log.warning("No IRNSS records found. Try an earlier --date.")
        sys.exit(0)

    df = records_to_csv(records, fetch_date)
    if df.empty:
        log.warning("No rows after processing.")
        sys.exit(0)

    df.to_csv(out_csv, index=False)
    log.info(f"Saved {len(df)} rows → {out_csv.name}")

    print("\n" + "─"*60)
    print(f"  Date    : {fetch_date.date()}")
    print(f"  Records : {len(df)}")
    for sat in sorted(df["satellite_id"].unique()):
        n = len(df[df["satellite_id"]==sat])
        print(f"    {sat}: {n} observations")
    print(f"  Output  : {out_csv}")
    print("─"*60)

if __name__ == "__main__":
    main()