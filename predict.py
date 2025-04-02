from pathlib import Path
import os
import torch
import numpy as np
import librosa
import subprocess
import tempfile
from model import LightingModel
from scipy.ndimage import gaussian_filter1d

def load_model(model_path, spect_dim=128, transformer_dim=1024, n_layers=6, in_channels=5):
    """
    Loads the trained LightingModel from a checkpoint.

    Args:
        model_path (str): Path to the saved model checkpoint.
        spect_dim (int): Spectrogram dimension.
        transformer_dim (int): Transformer dimension.
        n_layers (int): Number of transformer layers.
        in_channels (int): Number of input channels (spectrograms + beat frames).

    Returns:
        LightingModel: The loaded model.
    """
    model = LightingModel(spect_dim=spect_dim, transformer_dim=transformer_dim, 
                         n_layers=n_layers, in_channels=in_channels)
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

# Add a function to extract beat frames
def extract_beat_frames(audio_path, sr=22050, hop_length=512):
    """
    Extracts beat frames from audio using librosa (fallback method).
    
    Args:
        audio_path (Path or str): Path to the audio file.
        sr (int): Sampling rate.
        hop_length (int): Hop length for spectrogram.
    
    Returns:
        np.ndarray: Beat frames with shape [1, time].
    """
    print("Extracting beat frames...")
    
    try:
        # Try to use BeatThis if available
        from beat_this.inference import File2Beats
        print("Using BeatThis for beat detection")
        
        file2beats = File2Beats(
            checkpoint_path="final0", 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            dbn=False
        )
        
        # Get beat and downbeat times
        beats, downbeats = file2beats(audio_path)
        print(f"Found {len(beats)} beats and {len(downbeats)} downbeats")
        
        # Load audio to get duration and calculate total frames
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        total_frames = 1 + len(y) // hop_length
        audio_duration = len(y) / sr_loaded
        
        # Create beat activation frames
        beat_frames = np.zeros(total_frames)
        
        # Convert beat times to frame indices
        for beat_time in beats:
            if beat_time < audio_duration:
                frame_idx = int(beat_time * sr / hop_length)
                if frame_idx < total_frames:
                    beat_frames[frame_idx] = 0.8  # Regular beats
                    
        # Convert downbeat times to frame indices with higher activation
        for downbeat_time in downbeats:
            if downbeat_time < audio_duration:
                frame_idx = int(downbeat_time * sr / hop_length)
                if frame_idx < total_frames:
                    beat_frames[frame_idx] = 1.0  # Downbeats (stronger)
        
        # Apply smoothing to create slight ramp around beats
        beat_frames = gaussian_filter1d(beat_frames, sigma=1.0)
        
    except (ImportError, Exception) as e:
        print(f"Error using BeatThis: {e}. Falling back to librosa beat tracking.")
        
        # Fall back to librosa for beat detection
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        tempo, beat_frames_idx = librosa.beat.beat_track(y=y, sr=sr_loaded, hop_length=hop_length)
        
        # Create binary beat activation
        total_frames = 1 + len(y) // hop_length
        beat_frames = np.zeros(total_frames)
        beat_frames[beat_frames_idx] = 1.0
        
        # Apply smoothing
        beat_frames = gaussian_filter1d(beat_frames, sigma=1.0)
    
    return beat_frames.reshape(1, -1)

# Update combine_spectrograms function to include beat frames
def combine_spectrograms_with_beats(spectrograms_dict, beat_frames):
    """
    Combines individual spectrograms and beat frames into a single 5-channel spectrogram.

    Args:
        spectrograms_dict (dict): Dictionary mapping track names to spectrograms.
        beat_frames (np.ndarray): Beat frames with shape [1, time].

    Returns:
        np.ndarray: Combined 5-channel spectrogram with shape [5, n_mels, time].
    """
    # Define the order of tracks for consistent channel assignment
    track_order = ["drums", "bass", "vocals", "other"]
    
    # Get reference dimensions
    ref_spectrogram = next(iter(spectrograms_dict.values()))
    n_mels, time_len = ref_spectrogram.shape
    
    # Initialize combined spectrogram array (5 channels)
    combined = np.zeros((5, n_mels, time_len))
    
    # Fill in the tracks we have (first 4 channels)
    for i, track_name in enumerate(track_order):
        if track_name in spectrograms_dict:
            # Ensure all spectrograms have the same time dimension
            spec = spectrograms_dict[track_name]
            if spec.shape[1] != time_len:
                # Pad or truncate if necessary
                if spec.shape[1] > time_len:
                    spec = spec[:, :time_len]
                else:
                    pad_width = time_len - spec.shape[1]
                    spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
            
            combined[i] = spec
    
    # Add beat frames as 5th channel
    if beat_frames.shape[1] != time_len:
        # Resample beat frames to match the time dimension
        beat_frames = np.interp(
            np.linspace(0, beat_frames.shape[1]-1, time_len),
            np.arange(beat_frames.shape[1]),
            beat_frames[0]
        ).reshape(1, -1)
    
    # Broadcast beat frames across frequency dimension
    combined[4] = np.tile(beat_frames, (n_mels, 1))
    
    return combined

def predict_lighting(audio_path, model_path, output_folder, sr=22050, n_mels=128, hop_length=512):
    """
    Predicts lighting brightness values for a given audio file.

    Args:
        audio_path (Path): Path to the audio file.
        model_path (Path): Path to the trained model checkpoint.
        output_folder (Path): Folder to save the predictions.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for spectrogram.
    """
    audio_path = Path(audio_path)
    model_path = Path(model_path)
    output_folder = Path(output_folder)

    # Check if the audio file and model file exist
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the trained model with 5 input channels (4 spectrograms + beat frames)
    model = load_model(model_path, transformer_dim=1024, in_channels=5)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Extract beat frames
    beat_frames = extract_beat_frames(audio_path, sr, hop_length)

    # Create a temporary directory for source separation
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Separating audio tracks using HTDemucs...")
        separated_tracks = separate_audio_with_htdemucs(audio_path, temp_dir)
        
        # Extract spectrograms for each track
        track_spectrograms = {}
        for track_name, track_path in separated_tracks.items():
            if os.path.exists(track_path):
                print(f"Processing {track_name} track...")
                spectrogram = extract_audio_spectrogram(track_path, sr, n_mels, hop_length)
                track_spectrograms[track_name] = spectrogram
        
        # Create the combined 5-channel spectrogram (4 spectrograms + beat frames)
        if track_spectrograms:
            print("Creating combined 5-channel spectrogram...")
            multi_channel_spectrogram = combine_spectrograms_with_beats(track_spectrograms, beat_frames)
        else:
            # Fallback if source separation fails
            print("Warning: Source separation failed. Using fallback approach.")
            spectrogram = extract_audio_spectrogram(audio_path, sr, n_mels, hop_length)
            multi_channel_spectrogram = np.zeros((5, *spectrogram.shape))
            multi_channel_spectrogram[:4] = np.stack([spectrogram] * 4)
            multi_channel_spectrogram[4] = np.tile(beat_frames, (n_mels, 1))
    
    # Convert to tensor with shape [1, 5, freq, time]
    spectrogram_tensor = torch.tensor(multi_channel_spectrogram, dtype=torch.float32).unsqueeze(0)
    spectrogram_tensor = spectrogram_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

    # Predict brightness
    print("Generating brightness predictions...")
    with torch.no_grad():
        outputs = model(spectrogram_tensor)
        brightness = outputs.squeeze().cpu().numpy()

    # Save predictions
    output_folder.mkdir(parents=True, exist_ok=True)
    base_name = audio_path.stem
    prediction_file = output_folder / f"{base_name}_brightness.npy"
    np.save(prediction_file, brightness)

    print(f"Predictions saved to {prediction_file}")

# Update extract_audio_spectrogram to include hop_length
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
    app_path = path / "lumasync"
    audio_path = app_path / "stillcounting.wav"
    model_path = app_path / "trained_model1.pth"  # Path to the trained model checkpoint
    output_folder = app_path / "predictions"  # Folder to save the predictions

    predict_lighting(audio_path, model_path, output_folder)