"""
Downloads the CirCor DigiScope heart sound dataset (PhysioNet, CC BY 4.0)
from PhysioNet's open-data mirror on AWS, which is much faster than the zip.

Run:  python ml/download_data.py
"""

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "circor-heart-sound/1.0.3/"
OUT = Path(__file__).resolve().parent.parent / "data" / "circor-1.0.3"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_keys():
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={PREFIX}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(urllib.request.urlopen(url).read())
        keys += [c.find(NS + "Key").text for c in root.iter(NS + "Contents")]
        token_el = root.find(NS + "NextContinuationToken")
        if token_el is None:
            return keys
        token = token_el.text


def fetch(key):
    target = OUT / key[len(PREFIX):]
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(f"{BUCKET}/{key}").read()
    target.write_bytes(data)


def main():
    keys = [k for k in list_keys() if k.endswith((".wav", ".csv"))]
    print(f"Downloading {len(keys)} files to {OUT}")
    with ThreadPoolExecutor(32) as pool:
        for i, _ in enumerate(pool.map(fetch, keys), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(keys)}")
    print("Done")


if __name__ == "__main__":
    main()
