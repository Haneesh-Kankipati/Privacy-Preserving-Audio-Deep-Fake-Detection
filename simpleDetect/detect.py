import sys
import subprocess
import numpy as np
import torch
import librosa
import soundfile as sf

from model import AudioFakeDetector

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

# ⚠️ Set this correctly
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"


# =========================
# AUDIO LOADING (HYBRID)
# =========================
def load_audio(path):
    """
    Try soundfile first, fallback to FFmpeg if decoding fails
    """

    try:
        # ---- Attempt 1: soundfile ----
        waveform, sr = sf.read(path, dtype="float32")

        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, sr, TARGET_SR)

        return torch.from_numpy(waveform)

    except Exception as e:
        print("⚠️ soundfile failed, falling back to FFmpeg")
        # ---- Attempt 2: FFmpeg fallback ----

        cmd = [
            FFMPEG_PATH,
            "-i", path,
            "-ac", "1",
            "-ar", str(TARGET_SR),
            "-f", "f32le",
            "pipe:1"
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if proc.returncode != 0 or len(proc.stdout) == 0:
            raise RuntimeError(
                f"FFmpeg failed:\n{proc.stderr.decode(errors='ignore')}"
            )

        waveform = np.frombuffer(proc.stdout, dtype=np.float32)
        return torch.from_numpy(waveform)


# =========================
# DEEPFAKE DETECTION
# =========================
def detect_fake(audio_path):
    model = AudioFakeDetector().to(DEVICE)
    model.eval()

    waveform = load_audio(audio_path)

    # (1, time)
    waveform = waveform.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=1)

    return probs[0, 1].item()


# =========================
# CLI ENTRY
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    try:
        fake_probability = detect_fake(audio_path)

        print(f"\nFake Probability: {fake_probability:.2f}")

        if fake_probability >= 0.5:
            print("🟥 Audio is likely FAKE (AI-generated)")
        else:
            print("🟩 Audio is likely REAL")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
