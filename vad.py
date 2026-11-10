#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vad.py - Pembuang hening otomatis (Voice Activity Detection) untuk rekaman nonstop.

Dipakai SEBELUM jernihkan_audio.process_file:
  1. Baca seluruh file rekaman.
  2. Deteksi bagian yang berisi suara/aktivitas vs hening (noise floor).
  3. Sambung hanya bagian yang ada suaranya (dengan sedikit bantalan + fade).
  4. Tulis ke file WAV baru -> diteruskan ke pipeline pembersih biasa.

Metode: energi per-frame (RMS dB) pada sinyal yang sudah di-high-pass, dengan
ambang RELATIF terhadap noise floor file itu sendiri. Ini dipilih (bukan
librosa.effects.split yang memakai top_db relatif ke PUNCAK) karena rekaman
dalam tas sering punya hentakan keras sesaat (gesekan kain). Ambang relatif-
puncak akan salah menilai; ambang relatif-noise-floor jauh lebih stabil, dan
tidak butuh library tambahan (cukup numpy/scipy/soundfile yang sudah ada).
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

# preset "keberanian memotong" -> margin dB di atas noise floor
PRESET_MARGIN = {"aman": 6.0, "normal": 9.0, "agresif": 13.0}


def _frames_db(x, hop):
    n = len(x) // hop
    if n == 0:
        return np.array([-120.0])
    fr = x[: n * hop].reshape(n, hop)
    return 10.0 * np.log10(np.mean(fr * fr, axis=1, dtype=np.float64) + 1e-12)


def deteksi_segmen(
    mono,
    sr,
    margin_db=9.0,
    frame_ms=30.0,
    pad_ms=250.0,
    gap_ms=400.0,
    min_voice_ms=180.0,
    hpf_hz=150.0,
):
    """Kembalikan daftar [a, b] (indeks sampel) bagian yang ada suaranya."""
    # sinyal deteksi: high-pass buang gemuruh frekuensi rendah agar energi bersih
    sos = butter(4, hpf_hz / (sr / 2.0), btype="highpass", output="sos")
    pad = min(len(mono) - 1, int(0.2 * sr))
    xf = sosfiltfilt(sos, mono, padlen=max(1, pad)).astype(np.float32)

    hop = max(1, int(frame_ms / 1000.0 * sr))
    lvl = _frames_db(xf, hop)

    valid = lvl[lvl > -110.0]  # abaikan frame digital-silence (nol total)
    if len(valid) == 0:
        return [], dict(floor=-120.0, thr=-120.0)
    floor = float(np.percentile(valid, 10))  # noise floor = frame paling sunyi
    thr = floor + margin_db

    active = lvl > thr
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return [], dict(floor=floor, thr=thr)

    # kelompokkan frame aktif yang berurutan menjadi run [start, end]
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[brk + 1]])
    ends = np.concatenate([idx[brk], [idx[-1]]])

    # buang run yang terlalu pendek (klik/hentakan sesaat) SEBELUM diberi bantalan
    min_frames = max(1, int(min_voice_ms / 1000.0 * sr / hop))
    runs = [(s, e) for s, e in zip(starts, ends) if (e - s + 1) >= min_frames]
    if not runs:
        return [], dict(floor=floor, thr=thr)

    # ubah ke sampel + beri bantalan supaya awal/akhir kata tidak terpotong
    padn = int(pad_ms / 1000.0 * sr)
    segs = []
    for s, e in runs:
        a = max(0, s * hop - padn)
        b = min(len(mono), (e + 1) * hop + padn)
        segs.append([a, b])

    # gabungkan segmen yang jaraknya dekat (jeda antar kata/kalimat)
    gapn = int(gap_ms / 1000.0 * sr)
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a - merged[-1][1] <= gapn:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    return merged, dict(floor=floor, thr=thr)


def potong_hening(in_path, out_path, preset="normal", fade_ms=10.0, **kw):
    """
    Buang hening dari in_path, tulis hasil (yang sudah tersambung) ke out_path.
    Mengembalikan statistik. Jika tidak ada suara terdeteksi, file disalin apa
    adanya (fallback) dan kosong=True agar bisa diberi tahu ke pengguna.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    margin = PRESET_MARGIN.get(str(preset).lower(), 9.0)

    audio, sr = sf.read(str(in_path), dtype="float32", always_2d=True)
    total = len(audio)
    mono = audio.mean(axis=1)

    segs, info = deteksi_segmen(mono, sr, margin_db=margin, **kw)
    base = dict(sr=sr, dur_asli=total / sr, n_seg=len(segs),
                floor_db=round(info["floor"], 1), thr_db=round(info["thr"], 1))

    if not segs:
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
        base.update(dur_hasil=total / sr, kept=1.0, kosong=True)
        return base

    fade = int(fade_ms / 1000.0 * sr)
    pieces = []
    for a, b in segs:
        seg = audio[a:b].copy()
        n = min(fade, len(seg) // 2)
        if n > 0:  # fade kecil di tiap sambungan -> tanpa bunyi "klik"
            w = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            seg[:n] *= w
            seg[-n:] *= w[::-1]
        pieces.append(seg)

    out = np.concatenate(pieces, axis=0)
    sf.write(str(out_path), out, sr, subtype="PCM_16")
    base.update(dur_hasil=len(out) / sr, kept=len(out) / total, kosong=False)
    return base
