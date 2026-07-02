"""Vendor the West German reunification dataset as a long CSV.

Downloads the canonical Abadie, Diamond & Hainmueller (2015) reunification panel
-- West Germany plus 16 OECD donor countries observed annually 1960-2003 -- and
writes ``germany_reunification.csv`` next to this script. Run once to (re)create
the vendored fixture::

    python test/data/vendor_germany.py

Source
------
Abadie, A., Diamond, A., & Hainmueller, J. (2015). "Comparative Politics and the
Synthetic Control Method." *American Journal of Political Science*, 59(2),
495-510.

The dataset is the West German reunification panel used throughout the synthetic
-control and interactive-fixed-effects literature (it is the canonical second
benchmark after Proposition 99, and the demonstration dataset for Xu's ``gsynth``
interactive-fixed-effects package). We pull a faithful CSV export; the GDP values
match the published panel exactly (West Germany per-capita GDP is 2284 in 1960,
20465 in 1990, 28855 in 2003).

West Germany reunified in 1990, so it is treated from 1990 onward; the 16 other
OECD countries form the donor pool.

Layout written (one row per country-year):
    country   country name (treated unit = "West Germany")
    year      1960-2003 (integer)
    gdp       per-capita GDP, PPP-adjusted USD (the outcome)
    code, infrate, trade, schooling, invest60, invest70, invest80, industry
              the original Abadie predictors / identifiers, kept for provenance;
              the GeoCausality estimators use only ``gdp``
"""

import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/OscarEngelbrektson/"
    "SyntheticControlMethods/master/examples/datasets/german_reunification.csv"
)
OUT_PATH = Path(__file__).parent / "germany_reunification.csv"


def build_frame(raw_csv: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(raw_csv))
    df["year"] = df["year"].astype(int)
    return df.sort_values(["country", "year"]).reset_index(drop=True)


def main() -> None:
    with urllib.request.urlopen(DATA_URL) as resp:  # noqa: S310 (trusted source)
        raw = resp.read().decode("utf-8")
    df = build_frame(raw)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows ({df['country'].nunique()} countries) to {OUT_PATH}")


if __name__ == "__main__":
    main()
