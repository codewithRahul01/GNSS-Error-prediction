"""
Stage 0 — Auto-fetch daily IRNSS/NavIC broadcast ephemeris from NASA CDDIS
Parses RINEX 3.x navigation file → extracts I01/I04/I06 records
→ computes X,Y,Z,clock errors → saves CSV to Data/Raw/

Usage:
    python3 mlops/stage0_fetch.py                    # fetch today
    python3 mlops/stage0_fetch.py --date 2025-09-08  # fetch specific date
    python3 mlops/stage0_fetch.py --dry-run          # show URL, don't download

Env vars required:
    EARTHDATA_USER   NASA Earthdata username
    EARTHDATA_PASS   NASA Earthdata password
    (register free at https://urs.earthdata.nasa.gov)

Satellite mapping (IRNSS PRN → your pipeline IDs):
    I01 → GEO_A   (geostationary)
    I04 → MEO_B   (inclined geosynchronous)
    I06 → MEO_C   (inclined geosynchronous)
"""

import sys, os, re, gzip, math, struct, hashlib, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from urllib.request import Request, urlopen, HTTPPasswordMgrWithDefaultRealm, \
    HTTPBasicAuthHandler, build_opener, install_opener
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger("stage0")

# ── constants ────────────────────────────────────────────────────────────────
CDDIS_BASE   = "https://cddis.nasa.gov/archive/gnss/data/daily"
EARTHDATA_HOST = "urs.earthdata.nasa.gov"
CDDIS_HOST = "cddis.nasa.gov"
AUTH_REDIRECT_HOSTS = {CDDIS_HOST, EARTHDATA_HOST}

# IRNSS PRN → your satellite_id
SAT_MAP = {"I01": "GEO_A", "I04": "MEO_B", "I06": "MEO_C"}

# WGS-84 / IRNSS orbital constants
MU    = 3.986005e14   # m³/s²
OMEGA_E = 7.2921151467e-5  # rad/s
C     = 2.99792458e8  # m/s
F     = -4.442807633e-10  # relativistic correction factor

DATA_DIR = ROOT / "Data" / "Raw"

# ── auth setup ───────────────────────────────────────────────────────────────
def setup_auth():
    user = os.environ.get("EARTHDATA_USER", "")
    pwd  = os.environ.get("EARTHDATA_PASS", "")
    if not user or not pwd:
        log.error("Set EARTHDATA_USER and EARTHDATA_PASS env vars.")
        log.error("Register free at https://urs.earthdata.nasa.gov")
        sys.exit(1)
    return user, pwd

# ── URL builder ──────────────────────────────────────────────────────────────
def cddis_url(date: datetime) -> str:
    """
    Mixed broadcast ephemeris (includes IRNSS) daily file.
    Path: YYYY/DDD/YYp/BRDC00IGS_R_YYYYDDDHHMM_01D_MN.rnx.gz
    """
    doy  = date.timetuple().tm_yday        # day of year
    yr2  = date.strftime("%y")             # 2-digit year
    yr4  = date.strftime("%Y")
    fname = f"BRDC00IGS_R_{yr4}{doy:03d}0000_01D_MN.rnx.gz"
    return f"{CDDIS_BASE}/{yr4}/{doy:03d}/{yr2}p/{fname}", fname

# ── download ─────────────────────────────────────────────────────────────────
class CDDISSession(requests.Session):
    """
    Preserve Authorization only for redirects between CDDIS and Earthdata hosts.
    """
    def rebuild_auth(self, prepared_request, response):
        if "Authorization" not in prepared_request.headers:
            return

        original_host = (urlparse(response.request.url).hostname or "").lower()
        redirect_host = (urlparse(prepared_request.url).hostname or "").lower()
        if original_host in AUTH_REDIRECT_HOSTS and redirect_host in AUTH_REDIRECT_HOSTS:
            return

        return super().rebuild_auth(prepared_request, response)


def download(url: str, dest: Path, user: str, pwd: str) -> Path:
    """
    CDDIS uses cookie-based OAuth redirect via urs.earthdata.nasa.gov.
    Use a custom Session to preserve Basic Auth only across trusted redirects.
    """
    log.info(f"Downloading {url}")
    session = CDDISSession()
    session.auth = (user, pwd)

    try:
        resp = session.get(url, timeout=120, allow_redirects=True)

        # detect HTML auth error page
        if resp.headers.get("Content-Type", "").startswith("text/html"):
            log.error("Got HTML response — auth failed or file not found")
            log.error(f"HTTP {resp.status_code}: {url}")
            if resp.status_code in (401, 403):
                log.error("Check EARTHDATA_USER / EARTHDATA_PASS")
            elif resp.status_code == 404:
                log.error("File not yet available — CDDIS publishes ~24h after UTC midnight")
            else:
                log.error(resp.text[:300])
            sys.exit(1)

        resp.raise_for_status()
        dest.write_bytes(resp.content)
        log.info(f"Saved {len(resp.content)//1024} KB → {dest.name}")
        return dest

    except requests.RequestException as e:
        log.error(f"Network error: {e}")
        sys.exit(1)

# ── RINEX 3 parser (IRNSS nav) ───────────────────────────────────────────────
def parse_rinex_irnss(raw_text: str, date: datetime) -> list[dict]:
    """
    Parse RINEX 3 mixed navigation file.
    Keep only IRNSS (I**) records that fall within ±2 h of each clock hour
    on `date`.  Returns list of dicts with orbital elements.
    """
    target_sats = set(SAT_MAP.keys())
    records = []
    lines = raw_text.splitlines()

    # skip header
    i = 0
    while i < len(lines):
        if "END OF HEADER" in lines[i]:
            i += 1
            break
        i += 1

    while i < len(lines):
        line = lines[i]
        if len(line) < 23 or line[0] not in ("I",):
            i += 1
            continue
        prn = line[0:3].strip()
        if prn not in target_sats:
            i += 1
            continue

        # epoch line: PRN YYYY MM DD HH MM SS
        try:
            yr  = int(line[4:8])
            mo  = int(line[9:11])
            dy  = int(line[12:14])
            hr  = int(line[15:17])
            mn  = int(line[18:20])
            sc  = float(line[21:23])
            ep  = datetime(yr, mo, dy, hr, mn, int(sc), tzinfo=timezone.utc)
        except (ValueError, IndexError):
            i += 1
            continue

        # only keep records for our target date (UTC)
        if ep.date() != date.date():
            i += 1
            continue

        # clock params (broadcast 3 values on epoch line after epoch)
        try:
            a0 = float(line[23:42].replace("D","e").replace("d","e"))  # clock bias (s)
            a1 = float(line[42:61].replace("D","e").replace("d","e"))  # freq drift
            a2 = float(line[61:80].replace("D","e").replace("d","e"))  # freq rate
        except (ValueError, IndexError):
            i += 1
            continue

        # read next 7 broadcast record lines (56 chars per value, 4 vals/line)
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

        # unpack orbital elements (IRNSS RINEX record layout)
        # Broadcast record 1: IODE, Crs, delta_n, M0
        # Broadcast record 2: Cuc, e, Cus, sqrt_A
        # Broadcast record 3: toe, Cic, OMEGA0, Cis
        # Broadcast record 4: i0, Crc, omega, OMEGADOT
        # Broadcast record 5: IDOT, L2_codes, GPS_week, L2P_flag
        # Broadcast record 6: SV_acc, SV_health, TGD, IODC
        # Broadcast record 7: trans_time, fit_interval, ...
        try:
            Crs      = orb[1];  delta_n = orb[2];  M0    = orb[3]
            Cuc      = orb[4];  e       = orb[5];  Cus   = orb[6];  sqrtA = orb[7]
            toe      = orb[8];  Cic     = orb[9];  OMEGA0= orb[10]; Cis   = orb[11]
            i0       = orb[12]; Crc     = orb[13]; omega = orb[14]; OMEGADOT = orb[15]
            IDOT     = orb[16]
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

# ── orbital position + clock error ──────────────────────────────────────────
def kepler_solve(M: float, e: float, tol=1e-12) -> float:
    """Solve Kepler's equation M = E - e*sin(E) by Newton-Raphson."""
    E = M
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E

def compute_position_clock(rec: dict, t_utc: datetime) -> tuple[float, float, float, float]:
    """
    Given a broadcast nav record and a UTC time,
    compute ECEF X, Y, Z (meters) and clock error (meters).
    Returns (x, y, z, clock_err_m).
    """
    A = rec["sqrtA"] ** 2
    n0 = math.sqrt(MU / A**3)
    n  = n0 + rec["delta_n"]

    # time from epoch of ephemeris
    t  = (t_utc - rec["epoch"]).total_seconds()
    tk = t  # simplified: no week rollover for short windows

    M  = rec["M0"] + n * tk
    E  = kepler_solve(M, rec["e"])

    nu = math.atan2(
        math.sqrt(1 - rec["e"]**2) * math.sin(E),
        math.cos(E) - rec["e"]
    )

    phi = nu + rec["omega"]
    du  = rec["Cus"] * math.sin(2*phi) + rec["Cuc"] * math.cos(2*phi)
    dr  = rec["Crs"] * math.sin(2*phi) + rec["Crc"] * math.cos(2*phi)
    di  = rec["Cis"] * math.sin(2*phi) + rec["Cic"] * math.cos(2*phi)

    u = phi + du
    r = A * (1 - rec["e"] * math.cos(E)) + dr
    inc = rec["i0"] + di + rec["IDOT"] * tk

    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    OMEGA = rec["OMEGA0"] + (rec["OMEGADOT"] - OMEGA_E) * tk - OMEGA_E * rec["toe"]

    x = x_orb * math.cos(OMEGA) - y_orb * math.cos(inc) * math.sin(OMEGA)
    y = x_orb * math.sin(OMEGA) + y_orb * math.cos(inc) * math.cos(OMEGA)
    z = y_orb * math.sin(inc)

    # relativistic correction + clock polynomial
    dt_rel = F * rec["e"] * rec["sqrtA"] * math.sin(E)
    dt_sv  = rec["a0"] + rec["a1"] * tk + rec["a2"] * tk**2 + dt_rel
    clock_err_m = dt_sv * C   # seconds → meters

    return x, y, z, clock_err_m

# ── build error CSV ──────────────────────────────────────────────────────────
def records_to_csv(records: list[dict], date: datetime) -> pd.DataFrame:
    """
    For each record, evaluate position/clock at the epoch time.
    Compute 'error' as deviation from a simple mean position across the day
    (broadcast minus daily mean = relative broadcast error).
    """
    rows = []
    for rec in records:
        sat_id = SAT_MAP[rec["prn"]]
        x, y, z, clk = compute_position_clock(rec, rec["epoch"])
        rows.append({
            "satellite_id":       sat_id,
            "utc_time":           rec["epoch"].strftime("%Y-%m-%d %H:%M:%S"),
            "_x_abs": x, "_y_abs": y, "_z_abs": z,
            "satclockerror (m)":  round(clk, 6),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # compute error = position - daily mean (removes absolute orbit offset)
    for col, ecol in [("_x_abs","x_error (m)"),
                      ("_y_abs","y_error (m)"),
                      ("_z_abs","z_error (m)")]:
        means = df.groupby("satellite_id")[col].transform("mean")
        df[ecol] = (df[col] - means).round(6)

    df = df[["satellite_id","utc_time",
             "x_error (m)","y_error (m)","z_error (m)","satclockerror (m)"]]
    return df

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 0 — fetch IRNSS data from CDDIS")
    parser.add_argument("--date",    default=None,
        help="Date to fetch YYYY-MM-DD (default: yesterday UTC, safest for availability)")
    parser.add_argument("--dry-run", action="store_true",
        help="Print URL and exit without downloading")
    parser.add_argument("--force",   action="store_true",
        help="Re-download even if output CSV already exists")
    args = parser.parse_args()

    if args.date:
        fetch_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        # yesterday UTC — CDDIS files are typically available 12–24h after midnight UTC
        fetch_date = datetime.now(timezone.utc) - timedelta(days=1)

    url, fname = cddis_url(fetch_date)
    out_csv = DATA_DIR / f"irnss_{fetch_date.strftime('%Y-%m-%d')}.csv"

    log.info(f"Target date : {fetch_date.date()}")
    log.info(f"RINEX file  : {fname}")
    log.info(f"Output CSV  : {out_csv}")

    if args.dry_run:
        print(f"\nURL: {url}")
        print("Dry run — exiting.")
        return

    if out_csv.exists() and not args.force:
        log.info("CSV already exists. Use --force to re-fetch.")
        return

    user, pwd = setup_auth()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # download compressed RINEX
    tmp_gz = ROOT / "mlops" / "database" / fname
    download(url, tmp_gz, user, pwd)

    # decompress
    log.info("Decompressing…")
    with gzip.open(tmp_gz, "rt", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    tmp_gz.unlink()

    # parse
    records = parse_rinex_irnss(raw, fetch_date)
    if not records:
        log.warning("No IRNSS records found for target date. "
                    "Try --date with a date that has data.")
        sys.exit(0)

    # build CSV
    df = records_to_csv(records, fetch_date)
    if df.empty:
        log.warning("No rows after processing.")
        sys.exit(0)

    df.to_csv(out_csv, index=False)
    log.info(f"Saved {len(df)} rows → {out_csv.name}")

    # summary
    print("\n" + "─"*60)
    print(f"  Fetched : {fetch_date.date()}")
    print(f"  Records : {len(df)}")
    for sat in df["satellite_id"].unique():
        n = len(df[df["satellite_id"]==sat])
        print(f"    {sat}: {n} observations")
    print(f"  Output  : {out_csv}")
    print("─"*60)

if __name__ == "__main__":
    main()