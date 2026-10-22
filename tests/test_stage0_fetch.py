import gzip
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

import mlops.stage0_fetch as stage0_fetch


def _response(status, content, content_type, request_url, final_url=None):
    resp = requests.Response()
    resp.status_code = status
    resp._content = content
    resp.headers["Content-Type"] = content_type
    resp.url = final_url or request_url
    req = requests.PreparedRequest()
    req.prepare(method="GET", url=request_url)
    resp.request = req
    return resp


class Stage0FetchTests(unittest.TestCase):
    def test_cddis_url_for_2026_10_21(self):
        dt = stage0_fetch.datetime(2026, 10, 21, tzinfo=stage0_fetch.timezone.utc)
        url, fname = stage0_fetch.cddis_url(dt)
        self.assertEqual(fname, "BRDC00IGS_R_20262940000_01D_MN.rnx.gz")
        self.assertEqual(
            url,
            "https://cddis.nasa.gov/archive/gnss/data/daily/2026/294/26p/BRDC00IGS_R_20262940000_01D_MN.rnx.gz",
        )

    def test_download_accepts_valid_gzip_payload(self):
        url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/294/26p/file.gz"
        payload = gzip.compress(b"rinex-data")
        resp = _response(200, payload, "application/octet-stream", url)
        fake_session = mock.Mock()
        fake_session.get.return_value = resp
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "file.gz"
            with mock.patch.object(stage0_fetch, "_TrustedRedirectSession", return_value=fake_session):
                out = stage0_fetch.download(url, dest, "user", "pass")
            self.assertEqual(out, dest)
            self.assertEqual(dest.read_bytes(), payload)

    def test_download_rejects_html_auth_page(self):
        url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/294/26p/file.gz"
        content = b"<!DOCTYPE html><html><title>Earthdata Login</title></html>"
        resp = _response(200, content, "text/html; charset=UTF-8", url, "https://urs.earthdata.nasa.gov/oauth/authorize")
        fake_session = mock.Mock()
        fake_session.get.return_value = resp
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "file.gz"
            with mock.patch.object(stage0_fetch, "_TrustedRedirectSession", return_value=fake_session):
                with self.assertLogs("stage0", level="ERROR") as logs, self.assertRaises(SystemExit):
                    stage0_fetch.download(url, dest, "user", "pass")
        self.assertIn("Authentication failed", "\n".join(logs.output))

    def test_download_handles_missing_file(self):
        url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/294/26p/file.gz"
        resp = _response(404, b"Not Found", "text/html", url)
        fake_session = mock.Mock()
        fake_session.get.return_value = resp
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "file.gz"
            with mock.patch.object(stage0_fetch, "_TrustedRedirectSession", return_value=fake_session):
                with self.assertLogs("stage0", level="ERROR") as logs, self.assertRaises(SystemExit):
                    stage0_fetch.download(url, dest, "user", "pass")
        self.assertIn("RINEX file not found", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
