#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_app.py - Aplikasi web Penjernih Rekaman (Streamlit)

Urutan: (opsional) buang hening/VAD -> high-pass + noise reduction + EQ ->
kompresor + normalisasi + limiter. DSP sama persis dengan jernihkan_audio.py.

Deploy gratis di Streamlit Community Cloud (share.streamlit.io).
Keponakan cukup buka link, unggah WAV, unduh hasil. Semua dari HP.
"""

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import streamlit as st

from jernihkan_audio import process_file
from vad import potong_hening


def make_cfg(agresif_nr: bool) -> SimpleNamespace:
    return SimpleNamespace(
        hpf=180.0, hpf_order=6, boost=7.0, nr=0.90,
        agresif=agresif_nr, tanpa_nr=False, tanpa_kompresi=False, target=-18.0,
    )


def ensure_wav(src: Path) -> Path:
    """.WAV/.FLAC dipakai langsung; format lain dikonversi lewat ffmpeg."""
    if src.suffix.lower() in (".wav", ".flac"):
        try:
            sf.info(str(src))
            return src
        except Exception:
            pass
    out = src.with_suffix(".conv.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return out


def _mmss(detik: float) -> str:
    m, s = divmod(int(round(detik)), 60)
    return f"{m}:{s:02d}"


def proses_satu(upload, buang_hening, preset, agresif_nr):
    """Kembalikan (nama_hasil, data_bytes, pesan)."""
    cfg = make_cfg(agresif_nr)
    workdir = Path(tempfile.mkdtemp(prefix="st_"))
    src = workdir / upload.name
    src.write_bytes(upload.getbuffer())

    wav_in = ensure_wav(src)
    ket_vad = ""
    if buang_hening:
        trimmed = workdir / f"TRIM_{src.stem}.wav"
        stt = potong_hening(wav_in, trimmed, preset=preset)
        if stt["kosong"]:
            ket_vad = " · ⚠️ tak ada suara terdeteksi, hening tidak dibuang"
        else:
            ket_vad = (
                f" · 🔇 hening −{(1 - stt['kept']) * 100:.0f}% "
                f"({_mmss(stt['dur_asli'])}→{_mmss(stt['dur_hasil'])})"
            )
        wav_in = trimmed

    dst = workdir / f"JERNIH_{src.stem}.wav"
    r = process_file(wav_in, dst, cfg)
    pesan = f"✅ {upload.name}{ket_vad}"
    if r["clip_pct"] > 0.05:
        pesan += f" · ⚠️ {r['clip_pct']:.2f}% suara asli sudah pecah (GAIN alat terlalu tinggi)"
    return dst.name, dst.read_bytes(), pesan


st.set_page_config(page_title="Penjernih Rekaman", page_icon="🎧")
st.title("🎧 Penjernih Rekaman Suara")
st.write(
    "Unggah rekaman **.WAV**, tunggu sebentar, lalu **unduh** hasil yang sudah "
    "dibersihkan (desis dibuang, suara mendem diangkat, pelan dikeraskan, "
    "bagian hening dipotong). Bisa banyak file sekaligus."
)

files = st.file_uploader(
    "1) Pilih rekaman mentah (boleh banyak)", accept_multiple_files=True
)

with st.expander("⚙️ Pengaturan (boleh dibiarkan default)", expanded=True):
    buang_hening = st.checkbox(
        "🔇 Buang bagian hening otomatis (hilangkan jam-jam kosong)", value=True
    )
    preset = st.radio(
        "Seberapa berani memotong hening",
        ["aman", "normal", "agresif"], index=1, horizontal=True,
        help="Aman = potong sedikit (untuk obrolan pelan). "
        "Agresif = potong banyak (untuk rekaman yang sebagian besar sunyi).",
    )
    agresif_nr = st.checkbox(
        "🧹 Noise reduction ekstra (kalau desis masih tebal; kadang sedikit bergema)",
        value=False,
    )

if st.button("2) Jernihkan", type="primary"):
    if not files:
        st.warning("Unggah minimal satu file dulu.")
    else:
        hasil = []
        bar = st.progress(0.0)
        for i, up in enumerate(files, 1):
            with st.spinner(f"Membersihkan {up.name} …"):
                try:
                    nama, data, pesan = proses_satu(
                        up, buang_hening, preset, agresif_nr
                    )
                    hasil.append((nama, data))
                    st.write(pesan)
                except Exception as e:  # noqa: BLE001
                    st.write(f"❌ {up.name} GAGAL: {e}")
            bar.progress(i / len(files))
        st.session_state["hasil"] = hasil

if st.session_state.get("hasil"):
    st.subheader("3) Unduh hasil")
    for nama, data in st.session_state["hasil"]:
        st.download_button(
            f"⬇️ {nama}", data=data, file_name=nama,
            mime="audio/wav", key=nama,
        )
