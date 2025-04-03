import os
import librosa
import subprocess
import numpy as np
import cv2
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample
from seperate_drums import separate_drums_with_larsnet

def extract_audio_spectrogram(video_path, sr=22050, n_mels=128, hop_length=512):
    """
    Extracts the audio spectrogram from a video file.

    Args:
        video_path (str): Path to the video file.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for the spectrogram.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    # Load audio from video
    audio, _ = librosa.load(video_path, sr=sr)
    # Compute Mel spectrogram with specified hop_length
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    # Normalize the spectrogram
    log_spectrogram = (log_spectrogram - log_spectrogram.mean()) / log_spectrogram.std()
    return log_spectrogram

def extract_brightness(video_path):
    """
    Extracts brightness values from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: Brightness values for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    brightness_values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0  # Normalize brightness to [0, 1]
        brightness_values.append(brightness)
    cap.release()
    return brightness_values

def separate_audio_with_htdemucs(audio_path, output_dir):
    """
    Separates audio into individual tracks using HTDemucs.

    Args:
        audio_path (str): Path to the audio file.
        output_dir (str): Directory to save the separated tracks.

    Returns:
        dict: Paths to the separated tracks (e.g., drums, bass, vocals, etc.).
    """
    # Setup environment to avoid threading issues
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    
    # Run HTDemucs separation
    subprocess.run(["demucs", "-o", output_dir, audio_path], check=True, env=env)

    # Get the separated tracks
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    separated_dir = os.path.join(output_dir, "htdemucs", base_name)
    tracks = {
        "drums": os.path.join(separated_dir, "drums.wav"),
        "bass": os.path.join(separated_dir, "bass.wav"),
        "other": os.path.join(separated_dir, "other.wav"),
        "vocals": os.path.join(separated_dir, "vocals.wav"),
    }
    return tracks

def align_brightness_to_spectrogram(brightness, spectrogram_length):
    """
    Aligns the brightness vector to match the length of the spectrogram.

    Args:
        brightness (list or np.ndarray): The brightness vector.
        spectrogram_length (int): The number of time steps in the spectrogram.

    Returns:
        np.ndarray: Resampled brightness vector.
    """
    brightness = np.array(brightness)
    aligned_brightness = resample(brightness, spectrogram_length)
    return aligned_brightness

def preprocess_data(input_folder, output_folder, sr=22050, n_mels=128, hop_length=512):
    """
    Preprocesses all video files in the input folder, saving only:
    - Full audio spectrogram
    - Individual stem spectrograms (drums, bass, vocals, other)
    - Drums2 spectrogram (kick+snare+toms via LarsNet)
    - Brightness values

    Args:
        input_folder (str): Path to the folder containing video files.
        output_folder (str): Path to save the preprocessed data.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for the spectrogram.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Define track types to check
    track_types = ["drums", "bass", "vocals", "other"]
    
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith((".mov", ".mp4", ".mkv")):  # Support various video formats
            video_path = os.path.join(input_folder, file_name)
            base_name = os.path.splitext(file_name)[0]

            # Define the file paths we want to keep
            brightness_path = os.path.join(output_folder, f"{base_name}_brightness.npy")
            full_spec_path = os.path.join(output_folder, f"{base_name}_full_spectrogram.npy")
            audio_path = os.path.join(output_folder, f"{base_name}_audio.wav")
            drums2_spec_path = os.path.join(output_folder, f"{base_name}_drums2_spectrogram.npy")
            
            # Check if all separated spectrograms already exist
            separated_files_exist = True
            for track_name in track_types:
                track_spec_path = os.path.join(output_folder, f"{base_name}_{track_name}_spectrogram.npy")
                if not os.path.exists(track_spec_path):
                    separated_files_exist = False
                    break
            
            # Skip if all files already exist
            if (os.path.exists(brightness_path) and 
                os.path.exists(full_spec_path) and  
                os.path.exists(drums2_spec_path) and
                separated_files_exist):
                print(f"Skipping {file_name}: All required files already exist.")
                continue
            
            # Extract audio if needed
            if not os.path.exists(audio_path):
                print(f"Extracting audio from {file_name}")
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
                    "-ar", str(sr), "-ac", "1", audio_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Extract full audio spectrogram (non-separated, single channel)
            if not os.path.exists(full_spec_path) and os.path.exists(audio_path):
                print(f"Extracting full audio spectrogram for {file_name}")
                full_spectrogram = extract_audio_spectrogram(audio_path, sr, n_mels, hop_length)
                np.save(full_spec_path, full_spectrogram)
            
            # If we need to generate stem spectrograms, perform audio separation
            if not separated_files_exist:
                print(f"Extracting stem spectrograms for {file_name}")
                
                # Separate audio using HTDemucs
                separated_tracks = separate_audio_with_htdemucs(video_path, output_folder)

                # Extract spectrograms for each track
                for track_name, track_path in separated_tracks.items():
                    if os.path.exists(track_path):
                        # Check if this specific track spectrogram already exists
                        track_spec_path = os.path.join(output_folder, f"{base_name}_{track_name}_spectrogram.npy")
                        if not os.path.exists(track_spec_path):
                            # Generate new spectrogram with specified hop_length
                            spectrogram = extract_audio_spectrogram(track_path, sr, n_mels, hop_length)
                            np.save(track_spec_path, spectrogram)
                            print(f"Created {track_name} spectrogram at {track_spec_path}")
            
            # Process drums2 with LarsNet
            drums_path = os.path.join(
                os.path.join(output_folder, "htdemucs", base_name), 
                "drums.wav"
            )
            # Replace the drums2 processing section with this:
            if os.path.exists(drums_path) and not os.path.exists(drums2_spec_path):
                print(f"Processing drums with LarsNet for {file_name}")
                
                try:
                    # Use LarsNet to separate drum components
                    larsnet_waveform, waveform_sr = separate_drums_with_larsnet(
                        drum_audio_path=drums_path,
                        output_dir=output_folder,
                        wiener_filter=1.0,
                        device="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
                    )
                    
                    # Ensure the sample rate matches our target sample rate
                    if waveform_sr != sr:
                        print(f"Resampling drums2 audio from {waveform_sr}Hz to {sr}Hz to maintain consistent dimensions")
                        larsnet_waveform = librosa.resample(larsnet_waveform, orig_sr=waveform_sr, target_sr=sr)
                    
                    # Generate spectrogram with same parameters as other tracks
                    larsnet_spectrogram = librosa.feature.melspectrogram(
                        y=larsnet_waveform, sr=sr, n_mels=n_mels, hop_length=hop_length
                    )
                    log_larsnet_spectrogram = librosa.power_to_db(larsnet_spectrogram, ref=np.max)
                    # Normalize the spectrogram
                    log_larsnet_spectrogram = (log_larsnet_spectrogram - log_larsnet_spectrogram.mean()) / log_larsnet_spectrogram.std()

                    np.save(drums2_spec_path, log_larsnet_spectrogram)
                    print(f"Created drums2 spectrogram at {drums2_spec_path} with shape {log_larsnet_spectrogram.shape}")

                except Exception as e:
                    print(f"Error generating drums2 with LarsNet: {e}")
            
            # Extract brightness if needed
            if not os.path.exists(brightness_path):
                print(f"Extracting brightness for {file_name}")
                brightness = extract_brightness(video_path)
                
                # Use the full spectrogram for alignment reference if available
                if os.path.exists(full_spec_path):
                    full_spec = np.load(full_spec_path)
                    reference_length = full_spec.shape[-1]
                else:
                    # Use the first available track spectrogram
                    track_spec_path = os.path.join(output_folder, f"{base_name}_{track_types[0]}_spectrogram.npy")
                    if os.path.exists(track_spec_path):
                        track_spec = np.load(track_spec_path)
                        reference_length = track_spec.shape[-1]
                    else:
                        # Fallback if no spectrograms are available
                        reference_length = len(brightness)
                
                aligned_brightness = align_brightness_to_spectrogram(brightness, reference_length)
                np.save(brightness_path, aligned_brightness)

            print(f"Processed {file_name}")
            
    # Create a summary file listing all available preprocessed data
    create_dataset_summary(output_folder)

def create_dataset_summary(output_folder):
    """
    Creates a summary file listing all preprocessed data files.
    
    Args:
        output_folder (str): Path to the folder with preprocessed data.
    """
    summary = {
        "songs": [],
        "stats": {
            "total_songs": 0,
            "with_full_spectrogram": 0,
            "with_drums": 0,
            "with_bass": 0,
            "with_vocals": 0, 
            "with_other": 0,
            "with_drums2": 0,
            "with_brightness": 0
        }
    }
    
    # Get unique song names (without the suffixes)
    all_files = os.listdir(output_folder)
    all_files = [f for f in all_files if f.endswith('.npy')]
    
    # Extract base names
    base_names = set()
    for file in all_files:
        parts = file.split('_')
        if len(parts) > 1:
            base_name = '_'.join(parts[:-1])  # Remove the last part (spectrogram, brightness, etc.)
            if base_name.endswith(('_drums', '_bass', '_vocals', '_other', '_drums2')):
                base_name = '_'.join(base_name.split('_')[:-1])  # Remove stem type
            base_names.add(base_name)
    
    # For each song, check what data we have
    for base_name in sorted(base_names):
        song_data = {
            "name": base_name,
            "has_full_spectrogram": False,
            "has_drums": False,
            "has_bass": False,
            "has_vocals": False,
            "has_other": False,
            "has_drums2": False,
            "has_brightness": False
        }
        
        # Check for each type of data
        if os.path.exists(os.path.join(output_folder, f"{base_name}_full_spectrogram.npy")):
            song_data["has_full_spectrogram"] = True
            summary["stats"]["with_full_spectrogram"] += 1
            
        # Check stems individually
        stems = {
            "drums": "has_drums",
            "bass": "has_bass",
            "vocals": "has_vocals",
            "other": "has_other"
        }
        
        for stem, key in stems.items():
            stem_path = os.path.join(output_folder, f"{base_name}_{stem}_spectrogram.npy")
            if os.path.exists(stem_path):
                song_data[key] = True
                summary["stats"][f"with_{stem}"] += 1
        
        # Check for drums2
        if os.path.exists(os.path.join(output_folder, f"{base_name}_drums2_spectrogram.npy")):
            song_data["has_drums2"] = True
            summary["stats"]["with_drums2"] += 1
            
        # Check for brightness
        if os.path.exists(os.path.join(output_folder, f"{base_name}_brightness.npy")):
            song_data["has_brightness"] = True
            summary["stats"]["with_brightness"] += 1
        
        summary["songs"].append(song_data)
    
    summary["stats"]["total_songs"] = len(summary["songs"])
    
    # Save the summary as JSON
    import json
    with open(os.path.join(output_folder, "dataset_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary stats
    print(f"\nDataset Summary:")
    print(f"Total songs: {summary['stats']['total_songs']}")
    print(f"With full spectrogram: {summary['stats']['with_full_spectrogram']}")
    print(f"With drums: {summary['stats']['with_drums']}")
    print(f"With bass: {summary['stats']['with_bass']}")
    print(f"With vocals: {summary['stats']['with_vocals']}")
    print(f"With other: {summary['stats']['with_other']}")
    print(f"With drums2: {summary['stats']['with_drums2']}")
    print(f"With brightness data: {summary['stats']['with_brightness']}")

# Example usage
if __name__ == "__main__":
    path = Path(os.getcwd())
    preprocess_data(str(path / "data"), str(path / "preprocessed_data"), sr=22050, n_mels=128, hop_length=512)