#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jernihkan_audio.py - Penjernih batch rekaman voice recorder (suara teredam + bising)

Alur proses per file:
  1. High-pass filter 180 Hz      -> buang gemuruh mesin / ban / angin AC
  2. Spectral noise reduction     -> buang desis statis (profil noise diambil
                                     otomatis dari bagian paling sunyi)
  3. EQ vokal                     -> potong "mendem" 300 Hz, angkat 1-4 kHz
  4. Kompresor dynamic range      -> suara jauh/pelan ikut terangkat
  5. Normalisasi + limiter        -> lantang merata, dijamin tidak clipping

Pakai:
  python jernihkan_audio.py               (proses semua file baru)
  python jernihkan_audio.py --agresif     (noise reduction 2 tahap, lebih bersih)
  python jernihkan_audio.py --timpa       (proses ulang walau hasil sudah ada)
  python jernihkan_audio.py --help        (semua opsi)

Input : <folder skrip>/rekaman_mentah/*.WAV
Output: <folder skrip>/rekaman_jernih/CLEAR_<nama asli>.WAV
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import noisereduce as nr
    from scipy import signal
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
except ImportError as e:  # pragma: no cover
    print(f"[!] Library belum terpasang: {e.name}")
    print("    Jalankan:  pip install numpy scipy soundfile noisereduce")
    sys.exit(1)

# ----------------------------------------------------------------------------
# PENGATURAN BAWAAN (boleh diubah di sini, atau lewat opsi command line)
# ----------------------------------------------------------------------------
HPF_HZ = 180.0            # batas high-pass (150-200 Hz)
HPF_ORDER = 6             # kemiringan filter (6 = 36 dB/oktaf, zero-phase jadi 2x)
LPF_HZ = 7500.0           # buang desis di atas rentang vokal (dilewati jika sample rate rendah)

MUD_HZ, MUD_DB, MUD_Q = 300.0, -3.0, 1.0          # potong frekuensi "mendem"
PRES_HZ, PRES_DB, PRES_Q = 2200.0, 7.0, 0.7       # boost artikulasi ~1-4 kHz
AIR_HZ, AIR_DB, AIR_Q = 3500.0, 3.0, 1.2          # tambahan kejelasan konsonan

NR_STRENGTH = 0.90        # 0..1, seberapa banyak noise dibuang (tahap stasioner)
NR_STD_THRESH = 1.5       # ambang mask; lebih kecil = lebih agresif
NR_NOISE_FRACTION = 0.15  # porsi frame tersunyi yang dipakai sebagai profil noise
NR_NOISE_MAX_SEC = 8.0
NR_PASS2_STRENGTH = 0.60  # tahap kedua (non-stasioner), hanya dengan --agresif

COMP_THRESHOLD_DB = -30.0  # relatif terhadap level puncak vokal (lihat PRE_LEVEL_DB)
COMP_RATIO = 4.0
COMP_KNEE_DB = 8.0
COMP_ATTACK_MS = 8.0
COMP_RELEASE_MS = 250.0
PRE_LEVEL_DB = -14.0       # level bagian terkeras sebelum masuk kompresor

TARGET_RMS_DB = -18.0      # kelantangan akhir bagian yang ada suaranya
CEILING_DB = -1.0          # batas puncak mutlak (anti clipping)
LIMITER_MS = 5.0

CHUNK_SEC = 300            # file panjang diproses per potongan agar hemat RAM
CTX_SEC = 1.0              # konteks di tiap sisi potongan
XFADE_SEC = 0.25           # crossfade antar potongan

PREFIX = "CLEAR_"
DIR_IN = "rekaman_mentah"
DIR_OUT = "rekaman_jernih"


# ----------------------------------------------------------------------------
# BLOK DSP
# ----------------------------------------------------------------------------
def db2lin(db):
    return 10.0 ** (db / 20.0)


def peaking_sos(f0, gain_db, q, sr):
    """Biquad peaking EQ (RBJ cookbook) dalam bentuk SOS."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return np.hstack([b / den[0], den / den[0]])[None, :]


def build_filters(sr, cfg):
    nyq = sr / 2.0
    hpf = signal.butter(cfg.hpf_order, cfg.hpf / nyq, btype="highpass", output="sos")
    eq = []
    for f0, g, q in ((MUD_HZ, MUD_DB, MUD_Q),
                     (PRES_HZ, cfg.boost, PRES_Q),
                     (AIR_HZ, AIR_DB, AIR_Q)):
        if f0 < 0.45 * sr and abs(g) > 0.01:
            eq.append(peaking_sos(f0, g, q, sr))
    if LPF_HZ < 0.45 * sr:  # hanya berguna jika sample rate >= ~17 kHz
        eq.append(signal.butter(4, LPF_HZ / nyq, btype="lowpass", output="sos"))
    return hpf, (np.vstack(eq) if eq else None)


def frame_rms_db(x, hop):
    n = len(x) // hop
    if n == 0:
        return np.array([20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)])
    fr = x[: n * hop].reshape(n, hop)
    return 10.0 * np.log10(np.mean(fr * fr, axis=1, dtype=np.float64) + 1e-10)


def noise_profile(x, sr):
    """Ambil contoh noise dari frame-frame paling sunyi di potongan ini."""
    hop = int(0.25 * sr)
    lvl = frame_rms_db(x, hop)
    if len(lvl) < 4:
        return x
    order = np.argsort(lvl)
    # lewati frame digital-silence (nol total) yang tidak mewakili noise
    order = order[lvl[order] > -95.0]
    if len(order) == 0:
        return x
    k = max(2, int(len(order) * NR_NOISE_FRACTION))
    k = min(k, int(NR_NOISE_MAX_SEC / 0.25))
    idx = np.sort(order[:k])
    return np.concatenate([x[i * hop:(i + 1) * hop] for i in idx])


def denoise(x, sr, cfg):
    n_fft = int(2 ** round(np.log2(sr * 0.04)))  # ~40 ms: 512 @16k, 2048 @48k
    n_fft = int(np.clip(n_fft, 256, 4096))
    if len(x) < n_fft * 4:
        return x
    kw = dict(sr=sr, n_fft=n_fft, n_jobs=1)
    y = nr.reduce_noise(y=x, y_noise=noise_profile(x, sr), stationary=True,
                        prop_decrease=cfg.nr, n_std_thresh_stationary=NR_STD_THRESH,
                        **kw)
    if cfg.agresif:
        y = nr.reduce_noise(y=y, stationary=False, prop_decrease=NR_PASS2_STRENGTH, **kw)
    return np.nan_to_num(y).astype(np.float32, copy=False)


def clean_chunk(x, sr, hpf, eq, cfg):
    """Tahap 1-3 untuk satu potongan: HPF -> noise reduction -> EQ."""
    x = x - np.mean(x)                                   # buang DC offset
    pad = min(len(x) - 1, int(0.2 * sr))
    if pad < 30:
        return x.astype(np.float32)
    x = signal.sosfiltfilt(hpf, x, padlen=pad).astype(np.float32)
    if not cfg.tanpa_nr:
        x = denoise(x, sr, cfg)
    if eq is not None:
        x = signal.sosfiltfilt(eq, x, padlen=pad).astype(np.float32)
    return x


def apply_gain_curve(x, gain_blocks, hop):
    """Kalikan x dengan kurva gain per-blok (diinterpolasi), in-place & hemat RAM."""
    centers = (np.arange(len(gain_blocks)) + 0.5) * hop
    step = 2_000_000
    for s in range(0, len(x), step):
        e = min(len(x), s + step)
        g = np.interp(np.arange(s, e), centers, gain_blocks)
        x[s:e] *= g.astype(np.float32)
    return x


def compress(x, sr):
    """Kompresor feed-forward soft-knee; deteksi level per blok 5 ms."""
    hop = max(1, int(0.005 * sr))
    lvl = frame_rms_db(x, hop)
    over = lvl - COMP_THRESHOLD_DB
    k = COMP_KNEE_DB
    slope = 1.0 - 1.0 / COMP_RATIO
    gr = np.where(over <= -k / 2, 0.0,
                  np.where(over >= k / 2, slope * over,
                           slope * (over + k / 2) ** 2 / (2 * k)))
    target = -gr                                           # dB (<= 0)
    dt = hop / sr
    a_att = np.exp(-dt / (COMP_ATTACK_MS / 1000.0))
    a_rel = np.exp(-dt / (COMP_RELEASE_MS / 1000.0))
    out = np.empty_like(target)
    g = 0.0
    for i, t in enumerate(target):                         # smoothing attack/release
        a = a_att if t < g else a_rel
        g = a * g + (1.0 - a) * t
        out[i] = g
    return apply_gain_curve(x, db2lin(out), hop)


def speech_level_db(x, sr, pct):
    """Level (persentil) dari blok 50 ms yang aktif, abaikan bagian sunyi."""
    lvl = frame_rms_db(x, max(1, int(0.05 * sr)))
    active = lvl[lvl > lvl.max() - 45.0]
    return float(np.percentile(active if len(active) else lvl, pct))


def limiter(x, sr, ceiling):
    """Look-ahead limiter: puncak dijamin <= ceiling tanpa distorsi clipping."""
    win = max(3, int(LIMITER_MS / 1000.0 * sr) | 1)
    step = 4_000_000
    for s in range(0, len(x), step):
        a, b = max(0, s - win), min(len(x), s + step + win)
        seg = np.abs(x[a:b])
        need = np.minimum(1.0, ceiling / np.maximum(seg, 1e-9))
        need = minimum_filter1d(need, size=win, mode="nearest")
        need = uniform_filter1d(need, size=win, mode="nearest")
        e = min(len(x), s + step)
        x[s:e] *= need[s - a: s - a + (e - s)].astype(np.float32)
    np.clip(x, -ceiling, ceiling, out=x)                   # pengaman terakhir
    return x


# ----------------------------------------------------------------------------
# PROSES SATU FILE
# ----------------------------------------------------------------------------
def process_file(src, dst, cfg):
    with sf.SoundFile(str(src)) as f:
        sr, total, ch = f.samplerate, f.frames, f.channels
        if total < sr // 4:
            raise ValueError("rekaman terlalu pendek / kosong")
        hpf, eq = build_filters(sr, cfg)
        out = np.zeros(total, dtype=np.float32)
        chunk, ctx, xf = int(CHUNK_SEC * sr), int(CTX_SEC * sr), int(XFADE_SEC * sr)
        clipped = 0
        pos = 0
        while pos < total:
            end = min(total, pos + chunk)
            a = max(0, pos - ctx)
            b = min(total, end + ctx)
            f.seek(a)
            raw = f.read(b - a, dtype="float32", always_2d=True)
            clipped += int(np.sum(np.abs(raw[:, 0]) >= 0.999))
            mono = raw.mean(axis=1)                        # gabung kanal -> mono
            del raw
            y = clean_chunk(mono, sr, hpf, eq, cfg)
            if pos == 0:
                out[pos:end] = y[pos - a: end - a]
            else:                                          # crossfade dgn potongan sebelumnya
                n = min(xf, pos - a, end - pos)
                w = np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[pos:pos + n] = out[pos:pos + n] * (1 - w) + y[pos - a: pos - a + n] * w
                out[pos + n:end] = y[pos - a + n: end - a]
            if end < total:                                # sisakan ekor untuk crossfade berikut
                n = min(xf, b - end)
                out[end:end + n] = y[end - a: end - a + n]
            pos = end
            print(f"\r    membersihkan... {100 * pos // total:3d}%", end="", flush=True)
    print()

    # 4. Kompresi: samakan dulu level bagian terkeras, lalu tekan dynamic range
    out *= db2lin(PRE_LEVEL_DB - speech_level_db(out, sr, 95))
    if not cfg.tanpa_kompresi:
        out = compress(out, sr)

    # 5. Normalisasi kelantangan + limiter anti-clipping
    out *= db2lin(cfg.target - speech_level_db(out, sr, 70))
    out = limiter(out, sr, db2lin(CEILING_DB))

    tmp = dst.with_suffix(".tmp")
    sf.write(str(tmp), out, sr, subtype="PCM_16", format="WAV")
    tmp.replace(dst)
    return dict(sr=sr, ch=ch, dur=total / sr, peak=float(np.abs(out).max()),
                clip_pct=100.0 * clipped / total)


# ----------------------------------------------------------------------------
# BATCH
# ----------------------------------------------------------------------------
def find_wavs(folder):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".wav"
                  and not p.name.upper().startswith(PREFIX))


def fmt_dur(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main():
    ap = argparse.ArgumentParser(description="Penjernih batch rekaman voice recorder")
    base = Path(__file__).resolve().parent
    ap.add_argument("--input", type=Path, default=base / DIR_IN, help="folder sumber")
    ap.add_argument("--output", type=Path, default=base / DIR_OUT, help="folder hasil")
    ap.add_argument("--hpf", type=float, default=HPF_HZ, help="high-pass Hz (150-200)")
    ap.add_argument("--hpf-order", type=int, default=HPF_ORDER, help=argparse.SUPPRESS)
    ap.add_argument("--boost", type=float, default=PRES_DB, help="boost vokal 1-4 kHz (dB)")
    ap.add_argument("--nr", type=float, default=NR_STRENGTH, help="kekuatan noise reduction 0-1")
    ap.add_argument("--agresif", action="store_true", help="tambah NR tahap 2 (non-stasioner)")
    ap.add_argument("--target", type=float, default=TARGET_RMS_DB, help="kelantangan akhir dBFS")
    ap.add_argument("--tanpa-nr", action="store_true", help="lewati noise reduction")
    ap.add_argument("--tanpa-kompresi", action="store_true", help="lewati kompresor")
    ap.add_argument("--timpa", action="store_true", help="proses ulang file yang sudah ada")
    ap.add_argument("--ya", action="store_true", help="jawab otomatis 'ya' untuk semua pertanyaan")
    cfg = ap.parse_args()

    cfg.input.mkdir(parents=True, exist_ok=True)
    cfg.output.mkdir(parents=True, exist_ok=True)
    files = find_wavs(cfg.input)

    # Kemudahan: kalau folder mentah kosong tapi ada WAV di sebelah skrip, tawarkan pindah
    if not files:
        loose = find_wavs(base)
        if loose:
            print(f"Folder '{cfg.input.name}' kosong, tetapi ada {len(loose)} file WAV di {base}")
            ans = "y" if cfg.ya else input(f"Pindahkan ke '{cfg.input.name}' lalu proses? [Y/t] ")
            if ans.strip().lower() in ("", "y", "ya", "yes"):
                for p in loose:
                    shutil.move(str(p), str(cfg.input / p.name))
                files = find_wavs(cfg.input)
    if not files:
        print(f"Tidak ada file .WAV di: {cfg.input}")
        print("Taruh rekaman di folder itu lalu jalankan lagi.")
        return 0

    print(f"\n{len(files)} file ditemukan  |  HPF {cfg.hpf:.0f} Hz  |  boost {cfg.boost:+.0f} dB"
          f"  |  NR {cfg.nr:.2f}{' + agresif' if cfg.agresif else ''}\n")
    ok, skipped, failed = 0, 0, []
    t_all = time.time()
    for i, src in enumerate(files, 1):
        dst = cfg.output / f"{PREFIX}{src.name}"
        tag = f"[{i}/{len(files)}] {src.name}"
        if dst.exists() and not cfg.timpa:
            print(f"{tag}  -> sudah ada, dilewati")
            skipped += 1
            continue
        if src.stat().st_size < 1024:
            print(f"{tag}  -> file kosong, dilewati")
            failed.append((src.name, "file kosong (0 byte)"))
            continue
        print(f"{tag}  ({src.stat().st_size / 1e6:.1f} MB)")
        t0 = time.time()
        try:
            r = process_file(src, dst, cfg)
        except KeyboardInterrupt:
            print("\nDihentikan pengguna.")
            return 130
        except Exception as e:  # satu file rusak tidak menghentikan batch
            print(f"\n    GAGAL: {e}")
            failed.append((src.name, str(e)))
            continue
        ok += 1
        note = ""
        if r["clip_pct"] > 0.05:
            note = f"  | PERHATIAN: {r['clip_pct']:.2f}% sampel asli sudah pecah (gain terlalu tinggi)"
        print(f"    selesai {fmt_dur(r['dur'])} audio dalam {time.time() - t0:.0f} dtk"
              f"  | {r['sr']} Hz  | puncak {20 * np.log10(r['peak'] + 1e-9):.1f} dBFS{note}")

    print(f"\nRingkasan: {ok} berhasil, {skipped} dilewati, {len(failed)} gagal"
          f"  ({time.time() - t_all:.0f} dtk)")
    for name, why in failed:
        print(f"  - {name}: {why}")
    print(f"Hasil ada di: {cfg.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
