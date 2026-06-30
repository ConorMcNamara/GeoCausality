"""Vendor the Card & Krueger (1994) NJ/PA minimum-wage dataset as a long CSV.

Downloads David Card's public ``njmin`` data (the authors' own public release),
reshapes the two survey waves into the ``geo / date / y`` long format the
``EconometricEstimator`` subclasses expect, and writes ``card_krueger_1994.csv``
next to this script. Run once to (re)create the vendored fixture::

    python test/data/vendor_card_krueger.py

Source
------
Card, D., & Krueger, A. B. (1994). "Minimum Wages and Employment: A Case Study
of the Fast-Food Industry in New Jersey and Pennsylvania." *American Economic
Review*, 84(4), 772-793. Data: David Card's public data archive,
https://davidcard.berkeley.edu/data_sets/njmin.zip (file ``public.dat``, 410
restaurants), redistributed by the authors for replication.

Each restaurant ("store") was surveyed twice: wave 1 in Feb/Mar 1992 (before NJ
raised its minimum wage to $5.05 on 1 April 1992) and wave 2 in Nov/Dec 1992.
The outcome is full-time-equivalent employment,
``FTE = EMPFT + NMGRS + 0.5 * EMPPT`` (full-timers + managers + half of
part-timers), the exact definition used in the paper.

Reshaped layout (one row per store-wave with a non-missing FTE):
    geo   -- store id (``s<SHEET>``)
    date  -- 1992-02-01 (wave 1) or 1992-11-01 (wave 2)
    y     -- FTE employment
    state -- "NJ" (treated, min-wage rise) or "PA" (control)
"""

import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://davidcard.berkeley.edu/data_sets/njmin.zip"
OUT_PATH = Path(__file__).parent / "card_krueger_1994.csv"

# public.dat is whitespace-delimited with "." marking missing values; columns in
# codebook order.
COLUMNS = [
    "sheet",
    "chain",
    "co_owned",
    "state",
    "southj",
    "centralj",
    "northj",
    "pa1",
    "pa2",
    "shore",
    "ncalls",
    "empft",
    "emppt",
    "nmgrs",
    "wage_st",
    "inctime",
    "firstinc",
    "bonus",
    "pctaff",
    "meals",
    "open",
    "hrsopen",
    "psoda",
    "pfry",
    "pentree",
    "nregs",
    "nregs11",
    "type2",
    "status2",
    "date2",
    "ncalls2",
    "empft2",
    "emppt2",
    "nmgrs2",
    "wage_st2",
    "inctime2",
    "firstin2",
    "special2",
    "meals2",
    "open2r",
    "hrsopen2",
    "psoda2",
    "pfry2",
    "pentree2",
    "nregs2",
    "nregs112",
]
WAVE1_DATE = "1992-02-01"
WAVE2_DATE = "1992-11-01"


def _download_public_dat() -> list[str]:
    with urllib.request.urlopen(DATA_URL) as resp:  # noqa: S310 (trusted source)
        archive = zipfile.ZipFile(BytesIO(resp.read()))
    return archive.read("public.dat").decode("latin-1").splitlines()


def build_long_frame(lines: list[str]) -> pd.DataFrame:
    rows = [[np.nan if tok == "." else float(tok) for tok in ln.split()] for ln in lines]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["fte"] = df["empft"] + df["nmgrs"] + 0.5 * df["emppt"]
    df["fte2"] = df["empft2"] + df["nmgrs2"] + 0.5 * df["emppt2"]
    df["state_name"] = np.where(df["state"] == 1, "NJ", "PA")

    records = []
    for row in df.itertuples(index=False):
        geo = f"s{int(row.sheet)}"
        if not np.isnan(row.fte):
            records.append({"geo": geo, "date": WAVE1_DATE, "y": row.fte, "state": row.state_name})
        if not np.isnan(row.fte2):
            records.append({"geo": geo, "date": WAVE2_DATE, "y": row.fte2, "state": row.state_name})
    return pd.DataFrame.from_records(records)


def main() -> None:
    long_df = build_long_frame(_download_public_dat())
    long_df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(long_df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
