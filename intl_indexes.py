"""Build Yahoo ticker lists for major non-US index constituents (from Wikipedia)."""
import io
import re

import pandas as pd
import requests

H = {"User-Agent": "Mozilla/5.0"}
# name -> (url, suffix)
PAGES = {
    "KOSPI 200": ("https://en.wikipedia.org/wiki/KOSPI_200", [".KS", ".KQ"]),
    "FTSE 100": ("https://en.wikipedia.org/wiki/FTSE_100_Index", [".L"]),
    "FTSE 250": ("https://en.wikipedia.org/wiki/FTSE_250_Index", [".L"]),
    "DAX": ("https://en.wikipedia.org/wiki/DAX", [".DE"]),
    "MDAX": ("https://en.wikipedia.org/wiki/MDAX", [".DE"]),
    "CAC 40": ("https://en.wikipedia.org/wiki/CAC_40", [".PA"]),
    "AEX": ("https://en.wikipedia.org/wiki/AEX_index", [".AS"]),
    "IBEX 35": ("https://en.wikipedia.org/wiki/IBEX_35", [".MC"]),
    "SMI": ("https://en.wikipedia.org/wiki/Swiss_Market_Index", [".SW"]),
    "OMXS30": ("https://en.wikipedia.org/wiki/OMX_Stockholm_30", [".ST"]),
    "ASX 200": ("https://en.wikipedia.org/wiki/S%26P/ASX_200", [".AX"]),
    "Nifty 50": ("https://en.wikipedia.org/wiki/NIFTY_50", [".NS"]),
    "Nifty Next 50": ("https://en.wikipedia.org/wiki/NIFTY_Next_50", [".NS"]),
    "TSX 60": ("https://en.wikipedia.org/wiki/S%26P/TSX_60", [".TO"]),
    "EuroStoxx50": ("https://en.wikipedia.org/wiki/EURO_STOXX_50", [""]),
    "Hang Seng": ("https://en.wikipedia.org/wiki/Hang_Seng_Index", [".HK"]),
}


def build():
    out = {}
    for name, (url, sfxs) in PAGES.items():
        try:
            tabs = pd.read_html(io.StringIO(requests.get(url, headers=H, timeout=30).text))
        except Exception as e:
            print("skip", name, e)
            continue
        for t in tabs:
            col = next((c for c in t.columns
                        if any(w in str(c).lower() for w in ("ticker", "symbol", "code"))), None)
            if col is None or len(t) < 8:
                continue
            for v in t[col].astype(str):
                v = re.sub(r"^SEHK:\s*", "", v.replace("\xa0", " ")).strip()
                if not v or v == "nan":
                    continue
                if name == "Hang Seng":
                    v = v.zfill(4)
                if name == "KOSPI 200":
                    v = v.zfill(6)
                if name == "TSX 60":
                    v = v.replace(".", "-")
                for s in sfxs:
                    sym = v if "." in v and not s else v + s if "." not in v else v
                    out[sym.upper()] = name
            break
    return out


if __name__ == "__main__":
    d = build()
    print(len(d))
