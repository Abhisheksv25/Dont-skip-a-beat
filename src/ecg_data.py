"""
ecg_data.py -- loading and preprocessing for the MIT-BIH Arrhythmia Database.

Everything the project needs to go from PhysioNet files to a table of heartbeats:

    read_header()      parse a .hea header file (sampling rate, gain, lead names)
    read_signal()      decode a format-212 .dat signal file into millivolts
    read_annotations() decode a MIT-format .atr annotation file (beat locations + labels)
    load_record()      all three at once for one record
    extract_beats()    cut a fixed window around every annotated beat + RR-interval features

The readers are written with numpy only, so they run anywhere; they do the same job as
`wfdb.rdrecord` and `wfdb.rdann` from the official `wfdb` package. `verify_record()`
checks every decoded signal against the checksum and initial value stored in its own
header, and scripts/make_m1_figures.py also checks the beat counts against the database
documentation.
"""

from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

# --------------------------------------------------------------------------------------
# Record lists
# --------------------------------------------------------------------------------------

ALL_RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115,
               116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
               209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231,
               232, 233, 234]

# Records with a pacemaker. AAMI EC57 recommends excluding them.
PACED = [102, 104, 107, 217]

# Standard inter-patient split (de Chazal et al., 2004). DS1 = training pool, DS2 = test.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# --------------------------------------------------------------------------------------
# Annotation codes (WFDB ecgcodes.h) and the AAMI EC57 beat classes
# --------------------------------------------------------------------------------------

CODE2SYM = {
    1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E",
    11: "j", 12: "/", 13: "Q", 14: "~", 16: "|", 18: "s", 19: "T", 20: "*", 21: "D",
    22: '"', 23: "=", 24: "p", 25: "B", 26: "^", 27: "t", 28: "+", 29: "u", 30: "?",
    31: "!", 32: "[", 33: "]", 34: "e", 35: "n", 36: "@", 37: "x", 38: "f", 39: "(",
    40: ")", 41: "r",
}

# Symbols that mark a heartbeat (WFDB's isqrs set). Used for RR intervals.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?!")

# Original MIT-BIH label -> AAMI class
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",   # normal & bundle branch block
    "A": "S", "a": "S", "J": "S", "S": "S",             # supraventricular ectopic
    "V": "V", "E": "V",                                 # ventricular ectopic
    "F": "F",                                           # fusion of ventricular + normal
    "/": "Q", "f": "Q", "Q": "Q",                       # paced / unclassifiable
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_NAMES = {
    "N": "Normal / bundle branch block",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion",
    "Q": "Unknown / paced",
}

# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------

PHYSIONET_URL = "https://physionet.org/files/{db}/1.0.0/{name}"


def download_mitdb(dest: str, records=ALL_RECORDS) -> None:
    """Download the .hea, .dat and .atr files of each record from PhysioNet.

    Alternative: download the ZIP from https://physionet.org/content/mitdb/1.0.0/
    and unzip it; then point `dest` at the unzipped folder.
    """
    os.makedirs(dest, exist_ok=True)
    for rec in records:
        for ext in ("hea", "dat", "atr"):
            name = f"{rec}.{ext}"
            out = os.path.join(dest, name)
            if not os.path.exists(out):
                urllib.request.urlretrieve(PHYSIONET_URL.format(db="mitdb", name=name), out)


def find_record_dir(root: str) -> str:
    """Return the folder under `root` that contains 100.hea (handles unzipped layouts)."""
    for dirpath, _, files in os.walk(root):
        if "100.hea" in files:
            return dirpath
    raise FileNotFoundError(f"No MIT-BIH record (100.hea) found under {root}")


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------

@dataclass
class Header:
    record: str
    n_sig: int
    fs: float
    n_samples: int
    files: list = field(default_factory=list)
    formats: list = field(default_factory=list)
    gains: list = field(default_factory=list)
    baselines: list = field(default_factory=list)
    adc_res: list = field(default_factory=list)
    init_values: list = field(default_factory=list)
    checksums: list = field(default_factory=list)
    leads: list = field(default_factory=list)
    comments: list = field(default_factory=list)


def read_header(path_no_ext: str) -> Header:
    with open(path_no_ext + ".hea") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    comments = [ln[1:].strip() for ln in lines if ln.startswith("#")]
    lines = [ln for ln in lines if not ln.startswith("#")]
    rec, n_sig, fs, n_samp = lines[0].split()[:4]
    h = Header(record=rec, n_sig=int(n_sig), fs=float(fs.split("/")[0]),
               n_samples=int(n_samp), comments=comments)
    for ln in lines[1:1 + h.n_sig]:
        p = ln.split()
        m = re.match(r"([\d.]+)(?:\((-?\d+)\))?", p[2])
        adc_zero = int(p[4])
        h.files.append(p[0])
        h.formats.append(p[1])
        h.gains.append(float(m.group(1)))
        h.baselines.append(int(m.group(2)) if m.group(2) is not None else adc_zero)
        h.adc_res.append(int(p[3]))
        h.init_values.append(int(p[5]))
        h.checksums.append(int(p[6]))
        h.leads.append(" ".join(p[8:]) if len(p) > 8 else f"sig{len(h.leads)}")
    return h


def _decode_212(raw: bytes, n_sig: int, n_samples: int) -> np.ndarray:
    """Format 212: two 12-bit samples packed into three bytes."""
    b = np.frombuffer(raw, dtype=np.uint8)
    b = b[: (len(b) // 3) * 3].reshape(-1, 3).astype(np.int32)
    s0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s1 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    out = np.empty(2 * len(b), dtype=np.int32)
    out[0::2], out[1::2] = s0, s1
    out[out > 2047] -= 4096                     # 12-bit two's complement
    return out[: n_sig * n_samples].reshape(n_samples, n_sig)


def read_signal(path_no_ext: str, header: Header | None = None, physical: bool = True):
    """Return (signal [n_samples x n_sig], header). Physical units are millivolts."""
    h = header or read_header(path_no_ext)
    if set(h.formats) != {"212"}:
        raise NotImplementedError(f"Only format 212 is implemented, got {h.formats}")
    folder = os.path.dirname(path_no_ext)
    with open(os.path.join(folder, h.files[0]), "rb") as f:
        digital = _decode_212(f.read(), h.n_sig, h.n_samples)
    if not physical:
        return digital, h
    sig = (digital - np.array(h.baselines)) / np.array(h.gains)
    return sig.astype(np.float32), h


def read_annotations(path_no_ext: str, ext: str = "atr"):
    """Decode a MIT-format annotation file.

    Returns (samples, symbols, aux): sample index of each annotation, its label, and
    the auxiliary text (e.g. rhythm changes such as '(AFIB').
    """
    words = np.fromfile(f"{path_no_ext}.{ext}", dtype="<u2")
    t, i = 0, 0
    samples, symbols, aux = [], [], []
    while i < len(words):
        w = int(words[i])
        code, interval = w >> 10, w & 0x3FF
        if code == 0 and interval == 0:          # end of file
            break
        if code == 59:                           # SKIP: 32-bit interval follows
            hi, lo = int(words[i + 1]), int(words[i + 2])
            skip = (hi << 16) | lo
            if skip >= 2 ** 31:
                skip -= 2 ** 32
            t += skip
            i += 3
            continue
        if code == 63:                           # AUX: string of `interval` bytes follows
            n_words = (interval + 1) // 2
            text = words[i + 1: i + 1 + n_words].tobytes()[:interval]
            if aux:
                aux[-1] = text.decode("latin-1").rstrip("\x00")
            i += 1 + n_words
            continue
        if code in (60, 61, 62):                 # NUM / SUB / CHN modifiers
            i += 1
            continue
        t += interval
        samples.append(t)
        symbols.append(CODE2SYM.get(code, "?"))
        aux.append("")
        i += 1
    return np.array(samples), np.array(symbols), aux


@dataclass
class Record:
    name: str
    fs: float
    signal: np.ndarray        # n_samples x n_sig, mV
    leads: list
    ann_samples: np.ndarray
    ann_symbols: np.ndarray
    ann_aux: list
    header: Header


def load_record(folder: str, rec: int | str) -> Record:
    p = os.path.join(folder, str(rec))
    sig, h = read_signal(p)
    s, y, a = read_annotations(p)
    return Record(str(rec), h.fs, sig, h.leads, s, y, a, h)


def verify_record(folder: str, rec: int | str) -> dict:
    """Self-check: decoded samples must reproduce the header's initial value and checksum."""
    p = os.path.join(folder, str(rec))
    digital, h = read_signal(p, physical=False)
    ok_init = all(int(digital[0, k]) == h.init_values[k] for k in range(h.n_sig))
    ok_sum = True
    for k in range(h.n_sig):
        cs = int(digital[:, k].astype(np.int64).sum()) & 0xFFFF
        cs = cs - 65536 if cs >= 32768 else cs
        ok_sum &= cs == h.checksums[k]
    return {"record": str(rec), "initial_value_ok": ok_init, "checksum_ok": ok_sum}


# --------------------------------------------------------------------------------------
# Preprocessing and beat extraction
# --------------------------------------------------------------------------------------

def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 2):
    """Zero-phase Butterworth band-pass: removes baseline wander (<0.5 Hz) and
    high-frequency noise / mains interference (>40 Hz) without shifting the waveform."""
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def mlii_index(rec: Record) -> int:
    """Modified limb lead II is channel 0 in every record except 114 (channel 1)."""
    return rec.leads.index("MLII")


def extract_beats(rec: Record, pre: int = 90, post: int = 166, filtered: bool = True,
                  n_local: int = 10, long_window_s: float = 300.0):
    """Cut a window of `pre` samples before and `post` samples after each AAMI-labelled
    beat on lead MLII (90 + 166 = 256 samples = 0.71 s at 360 Hz).

    Returns a dict of arrays, one row per beat:
        X        beat waveforms (n_beats x 256), mV
        label    original MIT-BIH symbol
        aami     AAMI class (N/S/V/F/Q)
        sample   R-peak sample index
        rr_pre, rr_post, rr_local   RR intervals in seconds
        rr_ratio pre-RR divided by the mean of the previous `n_local` RR intervals
        rr_long  median RR interval in a +/- `long_window_s` window (label-free baseline)
        rr_ratio_long  pre-RR divided by rr_long
        prev_aami  AAMI class of the previous beat ('-' if it has none)
    """
    x = rec.signal[:, mlii_index(rec)]
    if filtered:
        x = bandpass(x, rec.fs)
    is_beat = np.array([s in BEAT_SYMBOLS for s in rec.ann_symbols])
    t = rec.ann_samples[is_beat]
    sym = rec.ann_symbols[is_beat]
    rr = np.diff(t) / rec.fs                     # rr[i] = t[i+1] - t[i]
    half = long_window_s * rec.fs

    rows = {k: [] for k in ["X", "label", "aami", "sample", "rr_pre", "rr_post",
                            "rr_local", "rr_ratio", "rr_long", "rr_ratio_long", "prev_aami"]}
    for i in range(1, len(t) - 1):               # need a previous and a next beat
        if sym[i] not in AAMI:
            continue
        if t[i] - pre < 0 or t[i] + post > len(x):
            continue
        rr_pre, rr_post = rr[i - 1], rr[i]
        prev = rr[max(0, i - 1 - n_local): i - 1]
        rr_local = prev.mean() if len(prev) else rr_pre
        lo, hi = np.searchsorted(t[:-1], [t[i] - half, t[i] + half])
        rr_long = np.median(rr[lo:hi])
        rows["X"].append(x[t[i] - pre: t[i] + post])
        rows["label"].append(sym[i])
        rows["aami"].append(AAMI[sym[i]])
        rows["sample"].append(t[i])
        rows["rr_pre"].append(rr_pre)
        rows["rr_post"].append(rr_post)
        rows["rr_local"].append(rr_local)
        rows["rr_ratio"].append(rr_pre / rr_local)
        rows["rr_long"].append(rr_long)
        rows["rr_ratio_long"].append(rr_pre / rr_long)
        rows["prev_aami"].append(AAMI.get(sym[i - 1], "-"))
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["X"] = out["X"].astype(np.float32)
    out["record"] = np.full(len(out["label"]), int(rec.name))
    return out


def beats_for_records(folder: str, records, **kw) -> dict:
    """Concatenate extract_beats() over several records."""
    parts = [extract_beats(load_record(folder, r), **kw) for r in records]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def aami_counts(folder: str, records) -> dict:
    """AAMI class counts per record, from the annotations only (no signal needed)."""
    table = {}
    for r in records:
        _, sym, _ = read_annotations(os.path.join(folder, str(r)))
        c = {k: 0 for k in AAMI_CLASSES}
        for s in sym:
            if s in AAMI:
                c[AAMI[s]] += 1
        table[r] = c
    return table
