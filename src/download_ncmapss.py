"""Robust resumable downloader for large files (N-CMAPSS 2.45GB) from hf-mirror."""
import os
import sys
import time
import urllib.request

URL = ("https://hf-mirror.com/datasets/NovaBenya/N-CMAPSS_DS02-006.h5/"
       "resolve/main/N-CMAPSS_DS02-006.h5")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "data", "raw", "N-CMAPSS_DS02-006.h5")
DEST = os.path.abspath(DEST)
CHUNK = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 100


def get_total(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0))


def download():
    total = get_total(URL)
    done = os.path.getsize(DEST) if os.path.exists(DEST) else 0
    print(f"total={total/1e6:.1f}MB resume_from={done/1e6:.1f}MB", flush=True)

    attempts = 0
    while done < total and attempts < MAX_RETRIES:
        attempts += 1
        headers = {"Range": f"bytes={done}-"}
        req = urllib.request.Request(URL, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(DEST, "ab") as f:
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    print(f"\r{done/1e6:.1f}/{total/1e6:.1f} MB "
                          f"({100*done/total:.1f}%)", end="", flush=True)
            print()
        except Exception as e:
            print(f"\n[retry {attempts}] {type(e).__name__}: {e}", flush=True)
            time.sleep(3)

    if done >= total:
        print(f"DOWNLOAD COMPLETE {done} bytes", flush=True)
        return 0
    print(f"INCOMPLETE: {done}/{total}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(download())
