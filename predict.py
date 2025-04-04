from pathlib import Path
import os
import torch
import numpy as np
import librosa
import subprocess
import tempfile
from model import LightingModel
from scipy.ndimage import gaussian_filter1d
from seperate_drums import separate_drums_with_larsnet

def load_model(
    model_path, 
    spect_dim=128, 
    transformer_dim=512, 
    n_layers=6, 
    in_channels=1,
    max_time_dim=None,
    transformer_chunk_size=None
):
    """
    Loads the trained LightingModel from a checkpoint.

    Args:
        model_path (str): Path to the saved model checkpoint.
        spect_dim (int): Spectrogram dimension.
        transformer_dim (int): Transformer dimension.
        n_layers (int): Number of transformer layers.
        in_channels (int): Number of input channels.
        max_time_dim (int): Max time dimension after downsampling. None to disable downsampling.
        transformer_chunk_size (int): Size of transformer chunks. None to disable chunking.

    Returns:
        LightingModel: The loaded model.
    """
    
    model = LightingModel(
        spect_dim=spect_dim, 
        transformer_dim=transformer_dim, 
        n_layers=n_layers, 
        in_channels=in_channels,
        max_time_dim=max_time_dim,
        transformer_chunk_size=transformer_chunk_size
    )
    
    checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

def separate_audio_with_htdemucs(audio_path, output_dir):
    """
    Separates audio into individual tracks using HTDemucs.

    Args:
        audio_path (str or Path): Path to the audio file.
        output_dir (str or Path): Directory to save the separated tracks.

    Returns:
        dict: Paths to the separated tracks (e.g., drums, bass, vocals, etc.).
    """
    # Run HTDemucs separation
    subprocess.run(["demucs", "-o", str(output_dir), str(audio_path)], check=True)

    # Get the separated tracks
    base_name = os.path.splitext(os.path.basename(str(audio_path)))[0]
    separated_dir = os.path.join(output_dir, "htdemucs", base_name)
    tracks = {
        "drums": os.path.join(separated_dir, "drums.wav"),
        "bass": os.path.join(separated_dir, "bass.wav"),
        "other": os.path.join(separated_dir, "other.wav"),
        "vocals": os.path.join(separated_dir, "vocals.wav"),
    }
    return tracks

def combine_spectrograms(spectrograms_dict, channels):
    """
    Combines spectrograms according to requested channels.

    Args:
        spectrograms_dict (dict): Dictionary mapping track names to spectrograms.
        channels (list): List of requested channels (e.g., ["full"] or ["drums", "bass"]).

    Returns:
        np.ndarray: Combined multi-channel spectrogram with shape [n_channels, n_mels, time].
    """
    # Get reference dimensions from first available spectrogram
    ref_spectrogram = next(iter(spectrograms_dict.values()))
    n_mels, time_len = ref_spectrogram.shape
    
    # Initialize list to collect spectrograms in order
    channel_spectrograms = []
    
    # Process requested channels
    for channel in channels:
        if channel in spectrograms_dict:
            # Get and process this channel's spectrogram
            spec = spectrograms_dict[channel]
            
            # Ensure consistent dimensions
            if spec.shape[1] != time_len:
                if spec.shape[1] > time_len:
                    spec = spec[:, :time_len]
                else:
                    pad_width = time_len - spec.shape[1]
                    spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
            
            # Add to our collection
            channel_spectrograms.append(spec)
    
    # Stack along first dimension to create multi-channel spectrogram
    if channel_spectrograms:
        return np.stack(channel_spectrograms)
    else:
        # Fallback: empty spectrogram
        return np.zeros((1, n_mels, time_len))

def predict_lighting(audio_path, model_path, output_folder, sr=22050, n_mels=128, 
                     hop_length=512, channels=["drums", "bass", "vocals", "other"],
                     transformer_dim=512, max_time_dim=None, transformer_chunk_size=None):
    """
    Predicts lighting brightness values for a given audio file.

    Args:
        audio_path (Path): Path to the audio file.
        model_path (Path): Path to the trained model checkpoint.
        output_folder (Path): Folder to save the predictions.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for spectrogram.
        channels (list): List of audio channels to use. Default is all stems.
                         Use "drums2" for LarsNet-processed drums.
    """
    audio_path = Path(audio_path)
    model_path = Path(model_path)
    output_folder = Path(output_folder)

    # Check if the audio file and model file exist
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Process audio to get spectrograms
    use_full = "full" in channels
    use_drums2 = "drums2" in channels
    stem_channels = [ch for ch in channels if ch != "full" and ch != "drums2"]
    
    all_spectrograms = {}
    
    if use_full:
        print("Processing full audio track...")
        full_spectrogram = extract_audio_spectrogram(audio_path, sr, n_mels, hop_length)
        all_spectrograms["full"] = full_spectrogram
    
    # Handle both regular stems and drums2 processing
    if stem_channels or use_drums2:
        print(f"Separating audio tracks using HTDemucs...")
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Need to extract drums if we need drums2
                stems_to_extract = stem_channels.copy()
                if use_drums2 and "drums" not in stems_to_extract:
                    stems_to_extract.append("drums")
                    
                separated_tracks = separate_audio_with_htdemucs(audio_path, temp_dir)
                
                # Process regular stem channels
                for track_name in stem_channels:
                    if track_name in separated_tracks and os.path.exists(separated_tracks[track_name]):
                        print(f"Processing {track_name} track...")
                        stem_spectrogram = extract_audio_spectrogram(
                            separated_tracks[track_name], sr, n_mels, hop_length
                        )
                        all_spectrograms[track_name] = stem_spectrogram
                    else:
                        print(f"Warning: {track_name} track not found or separation failed.")
                
                # Process drums2 using LarsNet if requested
                if use_drums2:
                    drums_path = separated_tracks.get("drums")
                    if drums_path and os.path.exists(drums_path):
                        print("Processing drums with LarsNet to generate drums2...")
                        try:
                            # Create a directory for LarsNet output within the temp dir
                            larsnet_output_dir = os.path.join(temp_dir, "larsnet_output")
                            os.makedirs(larsnet_output_dir, exist_ok=True)
                            
                            # Run LarsNet on the drums track
                            larsnet_waveform, waveform_sr = separate_drums_with_larsnet(
                                drum_audio_path=drums_path,
                                output_dir=larsnet_output_dir,
                                wiener_filter=1.0,
                                device="cuda:0" if torch.cuda.is_available() else "cpu"
                            )
                            
                            # Resample if necessary to match target sr
                            if waveform_sr != sr:
                                print(f"Resampling drums2 audio from {waveform_sr}Hz to {sr}Hz")
                                larsnet_waveform = librosa.resample(larsnet_waveform, orig_sr=waveform_sr, target_sr=sr)
                            
                            # Generate spectrogram from the waveform
                            print("Generating drums2 spectrogram...")
                            larsnet_spectrogram = librosa.feature.melspectrogram(
                                y=larsnet_waveform, sr=sr, n_mels=n_mels, hop_length=hop_length
                            )
                            log_larsnet_spectrogram = librosa.power_to_db(larsnet_spectrogram, ref=np.max)
                            # Normalize the spectrogram
                            log_larsnet_spectrogram = (log_larsnet_spectrogram - log_larsnet_spectrogram.mean()) / log_larsnet_spectrogram.std()
                            
                            all_spectrograms["drums2"] = log_larsnet_spectrogram
                            print(f"Drums2 spectrogram created with shape {log_larsnet_spectrogram.shape}")
                        except Exception as e:
                            print(f"Error generating drums2 with LarsNet: {e}")
                            print("Falling back to regular drums track for drums2...")
                            if "drums" in separated_tracks and os.path.exists(separated_tracks["drums"]):
                                drums_spectrogram = extract_audio_spectrogram(
                                    separated_tracks["drums"], sr, n_mels, hop_length
                                )
                                all_spectrograms["drums2"] = drums_spectrogram
                    else:
                        print("Warning: Drums track not found. Cannot generate drums2.")
                
            except Exception as e:
                print(f"Error during source separation: {e}")
                if not use_full:
                    print("Falling back to full audio spectrogram...")
                    full_spectrogram = extract_audio_spectrogram(audio_path, sr, n_mels, hop_length)
                    all_spectrograms["full"] = full_spectrogram
                    channels = ["full"]
    
    # Check if we have any spectrograms
    if not all_spectrograms:
        raise ValueError("Failed to extract any spectrograms from the audio.")
    
    # Process with available channels
    available_channels = list(all_spectrograms.keys())
    print(f"Available channels: {', '.join(available_channels)}")
    
    requested_available = [ch for ch in channels if ch in available_channels]
    if not requested_available:
        print("None of the requested channels are available. Using all available channels.")
        requested_available = available_channels
    
    print(f"Creating spectrogram with {len(requested_available)} channels: {', '.join(requested_available)}")
    multi_channel_spectrogram = combine_spectrograms(all_spectrograms, requested_available)
    
    # Load model without any limitations
    in_channels = multi_channel_spectrogram.shape[0]
    print(f"Loading model with {in_channels} input channels...")
    model = load_model(
        model_path, 
        in_channels=in_channels,
        transformer_dim=transformer_dim,
        max_time_dim=max_time_dim,
        transformer_chunk_size=transformer_chunk_size
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize the spectrogram
    for i in range(multi_channel_spectrogram.shape[0]):
        mean = np.mean(multi_channel_spectrogram[i])
        std = np.std(multi_channel_spectrogram[i]) + 1e-8
        multi_channel_spectrogram[i] = (multi_channel_spectrogram[i] - mean) / std
    
    # Convert to tensor
    spectrogram_tensor = torch.tensor(multi_channel_spectrogram, dtype=torch.float32).unsqueeze(0)
    spectrogram_tensor = spectrogram_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate predictions without limitations
    print("Generating brightness predictions...")
    with torch.no_grad():
        output = model(spectrogram_tensor)
        brightness = output.squeeze().cpu().numpy()
    
    # Save predictions
    output_folder.mkdir(parents=True, exist_ok=True)
    base_name = audio_path.stem
    channel_suffix = "_".join(requested_available)
    prediction_file = output_folder / f"{base_name}_brightness.npy"
    np.save(prediction_file, brightness)
    
    print(f"Final brightness shape: {brightness.shape}")
    print(f"Predictions saved to {prediction_file}")
    
    return prediction_file

def extract_audio_spectrogram(audio_path, sr=22050, n_mels=128, hop_length=512):
    """
    Extracts the audio spectrogram from an audio file.

    Args:
        audio_path (Path or str): Path to the audio file.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for spectrogram.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    audio_path = Path(audio_path) if not isinstance(audio_path, Path) else audio_path
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio, _ = librosa.load(str(audio_path), sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    # Normalize the spectrogram
    log_spectrogram = (log_spectrogram - log_spectrogram.mean()) / (log_spectrogram.std() + 1e-8)
    return log_spectrogram

# Example usage
if __name__ == "__main__":
    path = Path(os.getcwd())
    audio_path = path / "cloudsovercali.mp3"
    model_path = path / "best_model_drums2_bass_other_vocals_dim_1024_b8.pth"
    output_folder = path / "predictions"
    
    predict_lighting(
        audio_path, 
        model_path, 
        output_folder,
        channels=["drums2", "bass", "other", "vocals"],
        transformer_dim=1024,
        max_time_dim=None,
        transformer_chunk_size=None
    )
