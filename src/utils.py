import re

def ci_get(d: dict, key: str, default=None):
    # case-insensitive get
    key_l = key.lower()
    for k, v in d.items():
        if str(k).lower() == key_l:
            return v
    return default

def normalize_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def parse_percent_keys(cols):
    # Return a mapping for 'p5', 'p10', etc.
    percents = {}
    for c in cols:
        c_l = str(c).lower().strip()
        m = re.fullmatch(r"p(\d{1,2}|100)", c_l)
        if m:
            p = int(m.group(1))
            percents[p] = c
    return percents
