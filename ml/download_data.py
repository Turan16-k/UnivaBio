"""
Downloads PhysioNet datasets (open access) from PhysioNet's open-data mirror
on AWS, which is much faster than the zip download.

Run:  python ml/download_data.py              CirCor DigiScope (training data, ~550 MB)
      python ml/download_data.py --external   PhysioNet 2016 challenge (external test, ~200 MB)
"""

import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
DATA = Path(__file__).resolve().parent.parent / "data"
DATASETS = {
    "circor": ("circor-heart-sound/1.0.3/", DATA / "circor-1.0.3"),
    "external": ("challenge-2016/1.0.0/", DATA / "physionet2016"),
}
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_keys(prefix):
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(urllib.request.urlopen(url).read())
        keys += [c.find(NS + "Key").text for c in root.iter(NS + "Contents")]
        token_el = root.find(NS + "NextContinuationToken")
        if token_el is None:
            return keys
        token = token_el.text


def download(name):
    prefix, out = DATASETS[name]

    def fetch(key):
        target = out / key[len(prefix):]
        if target.exists() and target.stat().st_size > 0:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(6):
            try:
                target.write_bytes(urllib.request.urlopen(f"{BUCKET}/{key}").read())
                return
            except urllib.error.HTTPError as err:
                if err.code not in (500, 503) or attempt == 5:
                    raise
                time.sleep(2 ** attempt)  # the mirror throttles bursts of requests

    keys = [k for k in list_keys(prefix) if k.endswith((".wav", ".csv"))]
    if name == "external":
        keys = [k for k in keys if "/training-" in k]
    print(f"Downloading {len(keys)} files to {out}")
    with ThreadPoolExecutor(32) as pool:
        for i, _ in enumerate(pool.map(fetch, keys), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(keys)}")
    print("Done")


if __name__ == "__main__":
    download("external" if "--external" in sys.argv else "circor")
