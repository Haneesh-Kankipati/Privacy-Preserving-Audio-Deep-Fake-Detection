import torch
from pydub import AudioSegment
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

def load_flac_via_pydub(path, target_sr):
    # Convert FLAC to raw audio via pydub
    audio_seg = AudioSegment.from_file(path, format="flac")
    
    # Get samples as float32 numpy array in range [-1, 1]
    samples = audio_seg.get_array_of_samples()
    waveform = torch.tensor(samples, dtype=torch.float32)
    
    # Stereo handling
    if audio_seg.channels > 1:
        waveform = waveform.view(-1, audio_seg.channels).T  # (channels, samples)
    else:
        waveform = waveform.unsqueeze(0)  # mono
    
    # Resample if needed
    if audio_seg.frame_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=audio_seg.frame_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    return waveform

def audio_to_tokens(flac_path: str, device: str = "cpu"):
    # Load EnCodec model (24kHz)
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)  # kbps
    model.to(device)
    model.eval()

    # Load audio
    audio = load_flac_via_pydub(flac_path, model.sample_rate).to(device)

    # Add batch dimension
    if audio.dim() == 2:  # (channels, samples)
        audio = audio.unsqueeze(0)  # (1, channels, samples)

    with torch.no_grad():
        encoded_frames = model.encode(audio)

    # Extract discrete tokens
    tokens = torch.cat([frame[0] for frame in encoded_frames], dim=-1)
    return tokens.cpu()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert FLAC audio to acoustic tokens")
    parser.add_argument("audio_path", type=str, help="Path to .flac audio file")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    tokens = audio_to_tokens(args.audio_path, args.device)

    print("Token shape:", tokens.shape)
    print("First 20 tokens:")
    print(tokens[:, :20])
