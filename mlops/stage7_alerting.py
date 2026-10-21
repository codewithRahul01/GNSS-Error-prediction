"""
GNSS MLOps — Stage 7: Alerting + Auto-Retrain Loop
===================================================
Reads Stage 6 monitoring metrics, sends alerts when
thresholds are breached, and triggers Stage 4 retraining
when a model goes critical.

ALERT CHANNELS (configure in AlertConfig below)
------------------------------------------------
  1. Console  — always on, no setup needed
  2. Email    — set GNSS_EMAIL_FROM / GNSS_EMAIL_TO / GNSS_SMTP_*
  3. Slack    — set GNSS_SLACK_WEBHOOK env var

AUTO-RETRAIN LOOP
-----------------
  If a satellite's SW_W drops below SW_W_CRITICAL (0.70),
  Stage 7 triggers Stage 4 retraining automatically and
  logs the retrain result back to the database.

HOW TO RUN
----------
  # Run once right now:
  python mlops/stage7_alerting.py --mode run

  # Run and retrain any critical satellites:
  python mlops/stage7_alerting.py --mode run --auto-retrain

  # Show alert history:
  python mlops/stage7_alerting.py --mode history

  # Test alert channels (sends a test alert):
  python mlops/stage7_alerting.py --mode test

FILE LOCATION
-------------
  GNSS-Error-prediction/mlops/stage7_alerting.py

ENV VARS (optional — only needed for email/Slack)
-------------------------------------------------
  GNSS_EMAIL_FROM       sender email address
  GNSS_EMAIL_TO         recipient email (comma-separated for multiple)
  GNSS_SMTP_HOST        SMTP server host     (default: smtp.gmail.com)
  GNSS_SMTP_PORT        SMTP server port     (default: 587)
  GNSS_SMTP_USER        SMTP login username
  GNSS_SMTP_PASS        SMTP login password / app password
  GNSS_SLACK_WEBHOOK    Slack incoming webhook URL
"""

import argparse, logging, os, smtplib, sqlite3, sys, warnings
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib import request as urllib_request
import json

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "mlops" / "database" / "gnss.db"
LOG_DIR = ROOT / "mlops" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "stage7_alerting.log"),
    ],
)
log = logging.getLogger("STAGE7")

# ── Constants ────────────────────────────────────────────────
SATELLITES    = {"GEO_A": "GEO", "MEO_B": "MEO", "MEO_C": "MEO"}
SW_W_WARN     = 0.85
SW_W_CRITICAL = 0.70


# ════════════════════════════════════════════════════════════
# ALERT CONFIG — reads from environment variables
# ════════════════════════════════════════════════════════════

class AlertConfig:
    """
    Configuration for alert channels.
    All values read from env vars so credentials are never
    hardcoded in source.

    Set env vars in your shell or a .env file:
      export GNSS_EMAIL_TO="you@example.com"
      export GNSS_SMTP_PASS="your-app-password"
    """
    # Email
    email_enabled = bool(os.getenv("GNSS_EMAIL_TO"))
    email_from    = os.getenv("GNSS_EMAIL_FROM",  "gnss-monitor@isro.in")
    email_to      = os.getenv("GNSS_EMAIL_TO",    "").split(",")
    smtp_host     = os.getenv("GNSS_SMTP_HOST",   "smtp.gmail.com")
    smtp_port     = int(os.getenv("GNSS_SMTP_PORT", "587"))
    smtp_user     = os.getenv("GNSS_SMTP_USER",   "")
    smtp_pass     = os.getenv("GNSS_SMTP_PASS",   "")

    # Slack
    slack_enabled = bool(os.getenv("GNSS_SLACK_WEBHOOK"))
    slack_webhook = os.getenv("GNSS_SLACK_WEBHOOK", "")

    @classmethod
    def summary(cls):
        channels = ["console"]
        if cls.email_enabled: channels.append(f"email→{cls.email_to}")
        if cls.slack_enabled:  channels.append("slack")
        return channels


# ════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_alert_table():
    """Create alert_log table if it doesn't exist."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alert_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            alerted_at    TEXT    NOT NULL,
            satellite_id  TEXT    NOT NULL,
            severity      TEXT    NOT NULL,
            sw_w_avg      REAL,
            drift_detected INTEGER,
            alert_reason  TEXT,
            channels_used TEXT,
            retrain_triggered INTEGER DEFAULT 0,
            retrain_new_sw_w  REAL,
            retrain_promoted  INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_alert(satellite_id, severity, sw_w_avg, drift_detected,
              alert_reason, channels_used,
              retrain_triggered=False, retrain_new_sw_w=None, retrain_promoted=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO alert_log
            (alerted_at, satellite_id, severity, sw_w_avg, drift_detected,
             alert_reason, channels_used, retrain_triggered,
             retrain_new_sw_w, retrain_promoted)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        satellite_id, severity, sw_w_avg, int(drift_detected),
        alert_reason, ", ".join(channels_used),
        int(retrain_triggered), retrain_new_sw_w,
        int(retrain_promoted) if retrain_promoted is not None else None,
    ))
    conn.commit()
    conn.close()


def already_alerted_today(satellite_id: str, severity: str) -> bool:
    """
    Prevent duplicate alerts: return True if we already sent an
    alert of this severity for this satellite today.

    WHY: monitoring may run hourly. Without dedup, you'd get
    24 emails per day for a single stuck model.
    """
    conn  = get_conn()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row   = conn.execute("""
        SELECT id FROM alert_log
        WHERE satellite_id = ?
          AND severity = ?
          AND alerted_at LIKE ?
        LIMIT 1
    """, (satellite_id, severity, f"{today}%")).fetchone()
    conn.close()
    return row is not None


# ════════════════════════════════════════════════════════════
# ALERT MESSAGE BUILDER
# ════════════════════════════════════════════════════════════

def build_message(satellite_id: str, metrics: dict,
                  severity: str, retrain_result: dict = None) -> dict:
    """
    Build alert message text for all channels.
    Returns dict with 'subject', 'body_text', 'body_html', 'slack_blocks'.
    """
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sym      = "🔴" if severity == "critical" else "⚠️"
    sw_w     = metrics["sw_w_avg"]
    drift    = "YES" if metrics["drift_detected"] else "no"
    reason   = metrics.get("alert_reason", "")

    subject = (
        f"{sym} GNSS ALERT [{severity.upper()}] — "
        f"{satellite_id}  SW_W={sw_w:.4f}"
    )

    retrain_line = ""
    if retrain_result:
        new_sw   = retrain_result.get("sw_w_avg", 0)
        promoted = retrain_result.get("promoted", False)
        retrain_line = (
            f"\n🔄 AUTO-RETRAIN triggered\n"
            f"   New SW_W  : {new_sw:.4f}\n"
            f"   Promoted  : {'YES ✓' if promoted else 'NO (not better)'}"
        )

    body_text = f"""
GNSS Model Monitor — {ts}

Satellite  : {satellite_id}
Severity   : {severity.upper()}
SW_W avg   : {sw_w:.4f}  (warn≥0.85, critical≥0.70)
Drift      : {drift}
Reason     : {reason}
{retrain_line}

Benchmark  : SW_W = 0.9810 (SIH 2025 reference)
Action     : {'Model needs retraining' if severity == 'critical' else 'Monitor closely'}

-- GNSS MLOps Pipeline | SIH 2025 / ISRO
""".strip()

    body_html = f"""
<html><body style="font-family:monospace;background:#0d1117;color:#c9d1d9;padding:20px">
<h2 style="color:{'#f85149' if severity=='critical' else '#e3b341'}">{sym} GNSS Alert — {satellite_id}</h2>
<table style="border-collapse:collapse;width:100%">
  <tr><td style="padding:6px;color:#8b949e">Satellite</td><td style="padding:6px"><b>{satellite_id}</b></td></tr>
  <tr><td style="padding:6px;color:#8b949e">Severity</td><td style="padding:6px;color:{'#f85149' if severity=='critical' else '#e3b341'}"><b>{severity.upper()}</b></td></tr>
  <tr><td style="padding:6px;color:#8b949e">SW_W avg</td><td style="padding:6px"><b>{sw_w:.4f}</b></td></tr>
  <tr><td style="padding:6px;color:#8b949e">Drift</td><td style="padding:6px">{drift}</td></tr>
  <tr><td style="padding:6px;color:#8b949e">Reason</td><td style="padding:6px">{reason}</td></tr>
  <tr><td style="padding:6px;color:#8b949e">Timestamp</td><td style="padding:6px">{ts}</td></tr>
</table>
{'<p style="color:#58a6ff">🔄 Auto-retrain triggered. New SW_W: <b>' + str(round(retrain_result.get("sw_w_avg",0),4)) + '</b></p>' if retrain_result else ''}
<p style="color:#8b949e;font-size:12px">GNSS MLOps Pipeline — SIH 2025 / ISRO</p>
</body></html>
"""

    # Slack blocks (rich formatting)
    color  = "#f85149" if severity == "critical" else "#e3b341"
    slack_blocks = {
        "attachments": [{
            "color": color,
            "blocks": [
                {"type": "header", "text": {"type": "plain_text",
                 "text": f"{sym} GNSS Alert — {satellite_id} [{severity.upper()}]"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*Satellite:*\n{satellite_id}"},
                    {"type": "mrkdwn", "text": f"*SW_W avg:*\n{sw_w:.4f}"},
                    {"type": "mrkdwn", "text": f"*Severity:*\n{severity.upper()}"},
                    {"type": "mrkdwn", "text": f"*Drift:*\n{drift}"},
                ]},
                {"type": "section", "text": {"type": "mrkdwn",
                 "text": f"*Reason:* {reason}"}},
            ]
        }]
    }
    if retrain_result:
        slack_blocks["attachments"][0]["blocks"].append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f"🔄 *Auto-retrain triggered* — New SW_W: `{retrain_result.get('sw_w_avg',0):.4f}`  Promoted: {'✓' if retrain_result.get('promoted') else '✗'}"}
        })

    return {
        "subject"   : subject,
        "body_text" : body_text,
        "body_html" : body_html,
        "slack_blocks": slack_blocks,
    }


# ════════════════════════════════════════════════════════════
# CHANNEL SENDERS
# ════════════════════════════════════════════════════════════

def send_console(msg: dict, severity: str):
    """Always-on console alert — no config needed."""
    sym  = "🔴 CRITICAL" if severity == "critical" else "⚠️  WARNING"
    line = "─" * 55
    print(f"\n{line}")
    print(f"  {sym}")
    print(line)
    print(msg["body_text"])
    print(line + "\n")


def send_email(msg: dict) -> bool:
    """
    Send alert via SMTP email.
    Requires GNSS_EMAIL_TO and GNSS_SMTP_PASS env vars.

    For Gmail: create an App Password at
    myaccount.google.com/apppasswords (2FA must be on),
    then set GNSS_SMTP_PASS to that 16-char password.
    """
    cfg = AlertConfig
    try:
        email = MIMEMultipart("alternative")
        email["Subject"] = msg["subject"]
        email["From"]    = cfg.email_from
        email["To"]      = ", ".join(cfg.email_to)
        email.attach(MIMEText(msg["body_text"], "plain"))
        email.attach(MIMEText(msg["body_html"], "html"))

        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(cfg.smtp_user or cfg.email_from, cfg.smtp_pass)
            srv.sendmail(cfg.email_from, cfg.email_to, email.as_string())

        log.info("Email sent → %s", cfg.email_to)
        return True

    except Exception as e:
        log.error("Email failed: %s", e)
        return False


def send_slack(msg: dict) -> bool:
    """
    Send alert to a Slack channel via Incoming Webhook.
    Requires GNSS_SLACK_WEBHOOK env var.

    Setup:
      1. Slack → Your apps → Create/select app
      2. Features → Incoming Webhooks → Add New Webhook
      3. Copy the URL → export GNSS_SLACK_WEBHOOK="https://hooks.slack.com/..."
    """
    try:
        payload = json.dumps(msg["slack_blocks"]).encode("utf-8")
        req = urllib_request.Request(
            AlertConfig.slack_webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib_request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log.info("Slack alert sent")
                return True
            log.error("Slack returned %d", resp.status)
            return False
    except Exception as e:
        log.error("Slack failed: %s", e)
        return False


def dispatch_alert(satellite_id: str, metrics: dict,
                   severity: str,
                   retrain_result: dict = None) -> list:
    """
    Send alert on all configured channels.
    Returns list of channels actually used.
    """
    msg      = build_message(satellite_id, metrics, severity, retrain_result)
    used     = []

    send_console(msg, severity)
    used.append("console")

    if AlertConfig.email_enabled:
        if send_email(msg):
            used.append("email")

    if AlertConfig.slack_enabled:
        if send_slack(msg):
            used.append("slack")

    return used


# ════════════════════════════════════════════════════════════
# AUTO-RETRAIN TRIGGER
# ════════════════════════════════════════════════════════════

def trigger_retrain(satellite_id: str) -> dict:
    """
    Trigger Stage 4 retraining for a satellite.
    Called automatically when status == 'critical'.

    WHY auto-retrain on critical:
    A critical SW_W (<0.70) means the model's residuals are
    far from normal — the model is no longer reliable. Rather
    than waiting for a manual intervention, we retrain immediately
    using the latest data in the DB and promote if improved.
    """
    log.info("Triggering auto-retrain for %s...", satellite_id)
    from mlops.stage4_retrain import retrain
    result = retrain(satellite_id, n_restarts=8, force_promote=False)
    log.info("Retrain done: %s  SW_W=%.4f  promoted=%s",
             satellite_id, result["sw_w_avg"], result["promoted"])
    return result


# ════════════════════════════════════════════════════════════
# MAIN ALERT LOOP
# ════════════════════════════════════════════════════════════

def run_alert_loop(auto_retrain: bool = False) -> list:
    """
    Read latest monitoring metrics and send alerts as needed.

    Logic:
      status == 'ok'       → no alert
      status == 'warning'  → send warning alert (once per day)
      status == 'critical' → send critical alert + optionally retrain

    Returns list of alert dicts for logging/testing.
    """
    from mlops.stage6_monitoring import get_latest_metrics

    metrics_list = get_latest_metrics()
    if not metrics_list:
        log.warning("No monitoring data found. Run Stage 6 first.")
        print("  No monitoring data. Run: python mlops/stage6_monitoring.py --mode run")
        return []

    alerts_sent = []

    for metrics in metrics_list:
        sat      = metrics["satellite_id"]
        status   = metrics["status"]
        sw_w     = metrics["sw_w_avg"]

        if status == "ok":
            log.info("✓ %s  SW_W=%.4f  OK — no alert needed", sat, sw_w)
            continue

        severity = "critical" if status == "critical" else "warning"

        # Dedup — skip if we already alerted today at this level
        if already_alerted_today(sat, severity):
            log.info("⏭ %s  %s alert already sent today — skipping",
                     sat, severity)
            continue

        # Auto-retrain for critical models
        retrain_result = None
        if auto_retrain and severity == "critical":
            retrain_result = trigger_retrain(sat)

        # Dispatch to all channels
        channels = dispatch_alert(sat, metrics, severity, retrain_result)

        # Log to DB
        log_alert(
            satellite_id      = sat,
            severity          = severity,
            sw_w_avg          = sw_w,
            drift_detected    = metrics["drift_detected"],
            alert_reason      = metrics["alert_reason"],
            channels_used     = channels,
            retrain_triggered = retrain_result is not None,
            retrain_new_sw_w  = retrain_result["sw_w_avg"] if retrain_result else None,
            retrain_promoted  = retrain_result["promoted"]  if retrain_result else None,
        )

        alerts_sent.append({
            "satellite_id": sat,
            "severity"    : severity,
            "sw_w_avg"    : sw_w,
            "channels"    : channels,
            "retrained"   : retrain_result is not None,
        })

    return alerts_sent


# ════════════════════════════════════════════════════════════
# PUBLIC API — used by Stage 8 (dashboard) and cron triggers
# ════════════════════════════════════════════════════════════

def get_alert_history(limit: int = 20) -> list:
    """
    Return recent alert log entries.
    Called by Stage 8 dashboard to show alert timeline.
    """
    conn = get_conn()
    rows = conn.execute("""
        SELECT alerted_at, satellite_id, severity, sw_w_avg,
               drift_detected, alert_reason, channels_used,
               retrain_triggered, retrain_new_sw_w, retrain_promoted
        FROM alert_log
        ORDER BY alerted_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [
        {
            "alerted_at"       : r[0][:19],
            "satellite_id"     : r[1],
            "severity"         : r[2],
            "sw_w_avg"         : r[3],
            "drift_detected"   : bool(r[4]),
            "alert_reason"     : r[5],
            "channels_used"    : r[6],
            "retrain_triggered": bool(r[7]),
            "retrain_new_sw_w" : r[8],
            "retrain_promoted" : bool(r[9]) if r[9] is not None else None,
        }
        for r in rows
    ]


# ════════════════════════════════════════════════════════════
# MODES
# ════════════════════════════════════════════════════════════

def mode_run(auto_retrain: bool = False):
    init_alert_table()

    print("\n" + "=" * 58)
    print("  STAGE 7 — Alerting")
    print(f"  Channels: {', '.join(AlertConfig.summary())}")
    print("=" * 58 + "\n")

    alerts = run_alert_loop(auto_retrain=auto_retrain)

    if not alerts:
        print("  ✓ No new alerts to send (all OK or already alerted today)\n")
    else:
        print(f"\n  Sent {len(alerts)} alert(s):")
        for a in alerts:
            retrain_note = "  [retrained]" if a["retrained"] else ""
            print(f"    {a['severity'].upper():<10} {a['satellite_id']:<8} "
                  f"SW_W={a['sw_w_avg']:.4f}  "
                  f"via {', '.join(a['channels'])}{retrain_note}")

    mode_history(limit=5)


def mode_history(limit: int = 10):
    init_alert_table()
    history = get_alert_history(limit=limit)

    print("\n" + "─" * 72)
    print("  ALERT HISTORY")
    print("─" * 72)
    if not history:
        print("  No alerts sent yet.")
    else:
        for h in history:
            sym  = "🔴" if h["severity"] == "critical" else "⚠️ "
            ret  = " [retrained]" if h["retrain_triggered"] else ""
            print(f"  {sym} {h['alerted_at']}  {h['satellite_id']:<8}  "
                  f"SW_W={h['sw_w_avg']:.4f}  {h['severity']}{ret}")
            if h["alert_reason"]:
                print(f"       ↳ {h['alert_reason']}")
    print("─" * 72 + "\n")


def mode_test():
    """Send a test alert on all channels to verify configuration."""
    init_alert_table()
    print("\n" + "=" * 58)
    print("  STAGE 7 — Test Alert")
    print(f"  Channels: {', '.join(AlertConfig.summary())}")
    print("=" * 58 + "\n")

    fake_metrics = {
        "satellite_id"  : "TEST_SAT",
        "sw_w_avg"      : 0.6500,
        "drift_detected": True,
        "alert_reason"  : "TEST ALERT — please ignore",
    }
    channels = dispatch_alert("TEST_SAT", fake_metrics, "warning")
    print(f"\n  Test alert sent via: {', '.join(channels)}\n")


def mode_config():
    """Show current alert channel configuration."""
    cfg = AlertConfig
    print("\n" + "─" * 50)
    print("  STAGE 7 — Alert Config")
    print("─" * 50)
    print(f"  Email enabled : {cfg.email_enabled}")
    if cfg.email_enabled:
        print(f"  Email from    : {cfg.email_from}")
        print(f"  Email to      : {cfg.email_to}")
        print(f"  SMTP host     : {cfg.smtp_host}:{cfg.smtp_port}")
    else:
        print("  → Set GNSS_EMAIL_TO and GNSS_SMTP_PASS to enable")
    print()
    print(f"  Slack enabled : {cfg.slack_enabled}")
    if not cfg.slack_enabled:
        print("  → Set GNSS_SLACK_WEBHOOK to enable")
    print()
    print("  Console       : always enabled")
    print("─" * 50 + "\n")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GNSS MLOps Stage 7 — Alerting")
    parser.add_argument(
        "--mode", required=True,
        choices=["run", "history", "test", "config"],
        help="run=check+alert, history=show log, test=test channels, config=show config")
    parser.add_argument(
        "--auto-retrain", action="store_true",
        help="Automatically retrain critical satellites")
    parser.add_argument(
        "--limit", type=int, default=10,
        help="History rows to show (default: 10)")
    args = parser.parse_args()

    if args.mode == "run":
        mode_run(auto_retrain=args.auto_retrain)
    elif args.mode == "history":
        init_alert_table()
        mode_history(limit=args.limit)
    elif args.mode == "test":
        mode_test()
    elif args.mode == "config":
        mode_config()


if __name__ == "__main__":
    main()