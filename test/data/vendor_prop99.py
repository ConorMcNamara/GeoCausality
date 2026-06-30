"""Vendor the California Proposition 99 ("smoking") dataset as a long CSV.

Downloads the canonical Abadie, Diamond & Hainmueller (2010) Proposition 99
panel -- 39 states observed annually 1970-2000 -- and writes ``prop99_smoking.csv``
next to this script in the ``geo / date / y`` long format the synthetic-control
estimators expect. Run once to (re)create the vendored fixture::

    python test/data/vendor_prop99.py

Source
------
Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods
for Comparative Case Studies: Estimating the Effect of California's Tobacco
Control Program." *Journal of the American Statistical Association*, 105(490),
493-505.

The dataset is the ``smoking`` data distributed with the R ``Synth`` package
(Abadie, Diamond & Hainmueller), one of the most widely redistributed teaching
datasets in the synthetic-control literature. We pull a faithful CSV export of
it; the values match the package exactly (e.g. California cigarette sales are
123.0 packs in 1970, 90.1 in 1988, 41.6 in 2000).

California enacted Proposition 99 in 1988, so it is treated from 1989 onward;
the other 38 states (those without their own large tobacco-control programs in
the original study) are the donor pool.

Layout written (one row per state-year):
    state     state name (treated unit = "California")
    year      1970-2000 (integer)
    cigsale   per-capita cigarette sales, packs (the outcome)
    lnincome, beer, age15to24, retprice
              the original Abadie predictors, kept for provenance/completeness
              (our estimators use only ``cigsale``)
"""

import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/OscarEngelbrektson/"
    "SyntheticControlMethods/master/examples/datasets/smoking_data.csv"
)
OUT_PATH = Path(__file__).parent / "prop99_smoking.csv"


def build_frame(raw_csv: str) -> pd.DataFrame:
    from io import StringIO

    df = pd.read_csv(StringIO(raw_csv))
    df["year"] = df["year"].astype(int)
    return df.sort_values(["state", "year"]).reset_index(drop=True)


def main() -> None:
    with urllib.request.urlopen(DATA_URL) as resp:  # noqa: S310 (trusted source)
        raw = resp.read().decode("utf-8")
    df = build_frame(raw)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows ({df['state'].nunique()} states) to {OUT_PATH}")


if __name__ == "__main__":
    main()
