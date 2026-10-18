"""oracle_ofi.py - a slow, simple, independent rebuild of mdr_ofi in Python.

Why: the C++ engine is fast but clever (ArrayBook, windows, overflow).
This script uses the dumbest possible book (two dicts) and the CKS formula
written out again by hand. If both agree on thousands of real messages,
a bug would have to exist identically in two different programs.

usage: python scripts/oracle_ofi.py ~/md-replay/data/btc.evt data/ofi_20261001.csv [N]
"""
import sys
import numpy as np
import pandas as pd

EVT, OFI_CSV = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000   # messages to compare
DEPTH = 10

# --- read the binary file: 32-byte header, then 32-byte events (see event.hpp)
hdr = np.fromfile(EVT, dtype=np.dtype([("magic", "S8"), ("version", "<u2"), ("reserved", "<u2"),
                                       ("px_scale", "<u4"), ("qty_scale", "<u4"),
                                       ("count", "<u4"), ("min_book_px", "<i8")]), count=1)[0]
ev = np.fromfile(EVT, offset=32, dtype=np.dtype([("ts_exch", "<i8"), ("ts_local", "<i8"), ("px", "<i8"),
                                                 ("qty", "<i4"), ("side", "u1"), ("kind", "u1"),
                                                 ("flags", "u1"), ("reserved", "u1")]))
qs = float(hdr["qty_scale"])

bids, asks = {}, {}                          # price tick -> lots

def top(book, best_first_desc):
    """Best DEPTH levels as (px, qty); missing levels = worst price, size 0."""
    prices = sorted(book, reverse=best_first_desc)[:DEPTH]
    worst = -(2**63) if best_first_desc else 2**63 - 1
    lv = [(p, book[p]) for p in prices]
    return lv + [(worst, 0)] * (DEPTH - len(lv))

def ofi(bo, bn, ao, an):
    """CKS, written out again on purpose (not imported from anywhere)."""
    e_bid = (bn[1] if bn[0] >= bo[0] else 0) - (bo[1] if bn[0] <= bo[0] else 0)
    e_ask = (an[1] if an[0] <= ao[0] else 0) - (ao[1] if an[0] >= ao[0] else 0)
    return e_bid - e_ask

rows = []
prev = None
in_snap = have_snap = False
i, n = 0, len(ev)
while i < n and len(rows) < N:
    j = i                                     # one message = same ts_local
    while j < n and ev["ts_local"][j] == ev["ts_local"][i]:
        j += 1
    touched = snapshot = False
    for k in range(i, j):
        if ev["kind"][k] == 1:                # trade: book unchanged
            continue
        snap = bool(ev["flags"][k] & 1)
        if snap and not in_snap:              # new snapshot batch: wipe the book
            bids.clear(); asks.clear(); have_snap = True
        in_snap = snap
        if not have_snap:
            continue
        book = bids if ev["side"][k] == 0 else asks
        p, q = int(ev["px"][k]), int(ev["qty"][k])
        if q == 0: book.pop(p, None)
        else: book[p] = q
        touched = True
        snapshot |= snap
    i = j
    if not touched:
        continue
    if not bids or not asks or max(bids) >= min(asks):
        prev = None
        continue
    cur = (top(bids, True), top(asks, False))
    reset = snapshot or prev is None
    e = [0] * DEPTH if reset else [ofi(prev[0][m], cur[0][m], prev[1][m], cur[1][m]) for m in range(DEPTH)]
    rows.append(e + [int(reset)])
    prev = cur

py = pd.DataFrame(rows, columns=[f"e{m}" for m in range(1, DEPTH + 1)] + ["reset_flag"])
cpp = pd.read_csv(OFI_CSV, nrows=len(py))
cols = [f"e{m}" for m in range(1, DEPTH + 1)]
diff = (py[cols] / qs - cpp[cols]).abs().max().max()     # CSV prints %.8g, so allow rounding
flags_ok = (py["reset_flag"].values == cpp["reset_flag"].values).all()
bad = int(((py[cols] / qs - cpp[cols]).abs() > 1e-6).any(axis=1).sum())

print(f"messages compared: {len(py)}")
print(f"max |python - cpp| over e1..e10: {diff:.2e}")
print(f"rows that disagree: {bad}")
print(f"reset_flag identical: {flags_ok}")
print("ORACLE AGREES" if bad == 0 and flags_ok else "MISMATCH - investigate the first bad row")
