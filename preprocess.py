import os
import librosa
import subprocess
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample
import torch
from beat_this.inference import File2Beats

def extract_audio_spectrogram(video_path, sr=22050, n_mels=128):
    """
    Extracts the audio spectrogram from a video file.

    Args:
        video_path (str): Path to the video file.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    # Load audio from video
    audio, _ = librosa.load(video_path, sr=sr)
    # Compute Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
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
    # Run HTDemucs separation
    subprocess.run(["demucs", "-o", output_dir, audio_path], check=True)

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

def combine_spectrograms(spectrograms_dict):
    """
    Combines individual spectrograms into a single 4-channel spectrogram.

    Args:
        spectrograms_dict (dict): Dictionary mapping track names to spectrograms.

    Returns:
        np.ndarray: Combined 4-channel spectrogram with shape [4, n_mels, time].
    """
    # Define the order of tracks for consistent channel assignment
    track_order = ["drums", "bass", "vocals", "other"]
    
    # Get reference dimensions
    ref_spectrogram = next(iter(spectrograms_dict.values()))
    n_mels, time_len = ref_spectrogram.shape
    
    # Initialize combined spectrogram array
    combined = np.zeros((4, n_mels, time_len))
    
    # Fill in the tracks we have
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
    
    return combined

def extract_beat_frames(audio_path, sr=22050, n_mels=128, hop_length=512):
    """
    Extracts beat frames from an audio file using BeatThis.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length for the spectrogram.

    Returns:
        np.ndarray: Beat frames with shape [1, time].
    """
    try:
        # Initialize BeatThis
        print(f"Using BeatThis to extract beats from {os.path.basename(audio_path)}")
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
        from scipy.ndimage import gaussian_filter1d
        beat_frames = gaussian_filter1d(beat_frames, sigma=1.0)
        
        return beat_frames.reshape(1, -1)
        
    except (ImportError, ModuleNotFoundError, Exception) as e:
        print(f"Error using BeatThis: {str(e)}. Falling back to librosa beat tracking.")
        # Fall back to librosa for beat detection
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        tempo, beat_frames_idx = librosa.beat.beat_track(y=y, sr=sr_loaded, hop_length=hop_length)
        
        # Create binary beat activation
        total_frames = 1 + len(y) // hop_length
        beat_frames = np.zeros(total_frames)
        beat_frames[beat_frames_idx] = 1.0
        
        # Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        beat_frames = gaussian_filter1d(beat_frames, sigma=1.0)
        
        return beat_frames.reshape(1, -1)

def preprocess_data(input_folder, output_folder, sr=22050, n_mels=128, hop_length=512):
    """
    Preprocesses all video files in the input folder, saving:
    - Full audio spectrogram (single-channel)
    - Individual stem spectrograms (drums, bass, vocals, other)
    - Combined 4-channel spectrogram
    - Combined 5-channel spectrogram with beat frames
    - Beat frames
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

            # Define all file paths
            brightness_path = os.path.join(output_folder, f"{base_name}_brightness.npy")
            combined_spec_path = os.path.join(output_folder, f"{base_name}_combined_spectrogram.npy")
            full_spec_path = os.path.join(output_folder, f"{base_name}_full_spectrogram.npy")  # New: full audio spectrogram
            beat_frames_path = os.path.join(output_folder, f"{base_name}_beatframes.npy")
            audio_path = os.path.join(output_folder, f"{base_name}_audio.wav")
            combined_5ch_path = os.path.join(output_folder, f"{base_name}_combined_5ch_spectrogram.npy")
            
            # Check if all separated spectrograms already exist
            separated_files_exist = True
            for track_name in track_types:
                track_spec_path = os.path.join(output_folder, f"{base_name}_{track_name}_spectrogram.npy")
                if not os.path.exists(track_spec_path):
                    separated_files_exist = False
                    break
            
            # Skip if all files already exist (including full spectrogram)
            if (os.path.exists(brightness_path) and 
                os.path.exists(combined_spec_path) and 
                os.path.exists(beat_frames_path) and
                os.path.exists(combined_5ch_path) and
                os.path.exists(full_spec_path) and  # Also check for full spectrogram
                separated_files_exist):
                print(f"Skipping {file_name}: All preprocessed files already exist.")
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
            
            # Extract beat frames if needed
            if not os.path.exists(beat_frames_path):
                print(f"Extracting beat frames for {file_name}")
                beat_frames = extract_beat_frames(audio_path, sr, n_mels, hop_length)
                np.save(beat_frames_path, beat_frames)
            else:
                beat_frames = np.load(beat_frames_path)
            
            # Process brightness if needed but spectrograms exist
            if separated_files_exist and os.path.exists(combined_spec_path) and not os.path.exists(brightness_path):
                print(f"Only extracting brightness for {file_name}")
                # Load a spectrogram to get its length
                reference_spec = np.load(os.path.join(output_folder, f"{base_name}_{track_types[0]}_spectrogram.npy"))
                reference_length = reference_spec.shape[-1]
                
                # Extract and save brightness
                brightness = extract_brightness(video_path)
                aligned_brightness = align_brightness_to_spectrogram(brightness, reference_length)
                np.save(brightness_path, aligned_brightness)
            
            # If we need to generate spectrograms, perform audio separation
            if not separated_files_exist or not os.path.exists(combined_spec_path):
                print(f"Extracting stem spectrograms for {file_name}")
                
                # Separate audio using HTDemucs
                separated_tracks = separate_audio_with_htdemucs(video_path, output_folder)

                # Extract spectrograms for each track
                track_spectrograms = {}
                for track_name, track_path in separated_tracks.items():
                    if os.path.exists(track_path):
                        # Check if this specific track spectrogram already exists
                        track_spec_path = os.path.join(output_folder, f"{base_name}_{track_name}_spectrogram.npy")
                        if os.path.exists(track_spec_path):
                            # Load existing spectrogram
                            spectrogram = np.load(track_spec_path)
                        else:
                            # Generate new spectrogram with specified hop_length
                            spectrogram = extract_audio_spectrogram(track_path, sr, n_mels, hop_length)
                            np.save(track_spec_path, spectrogram)
                        
                        track_spectrograms[track_name] = spectrogram

                # Create and save the combined 4-channel spectrogram
                if track_spectrograms:
                    combined_spectrogram = combine_spectrograms(track_spectrograms)
                    np.save(combined_spec_path, combined_spectrogram)
                    
                    # Create the 5-channel spectrogram with beat frames
                    if os.path.exists(beat_frames_path):
                        beat_frames = np.load(beat_frames_path)
                        # Make sure beat frames match the spectrogram time dimension
                        if beat_frames.shape[1] != combined_spectrogram.shape[2]:
                            beat_frames = np.interp(
                                np.linspace(0, beat_frames.shape[1]-1, combined_spectrogram.shape[2]),
                                np.arange(beat_frames.shape[1]),
                                beat_frames[0]
                            ).reshape(1, -1)
                            # Save the resampled beat frames
                            np.save(beat_frames_path, beat_frames)
                        
                        # Create 5-channel spectrogram (4 audio tracks + beat frames)
                        combined_5ch = np.zeros((5, *combined_spectrogram.shape[1:]))
                        combined_5ch[:4] = combined_spectrogram
                        # Expand beat frames to match frequency dimension
                        beat_expanded = np.tile(beat_frames, (n_mels, 1))
                        combined_5ch[4] = beat_expanded
                        np.save(combined_5ch_path, combined_5ch)

            # Extract brightness if needed
            if not os.path.exists(brightness_path):
                print(f"Extracting brightness for {file_name}")
                brightness = extract_brightness(video_path)
                
                # If we already have spectrograms, use them for alignment reference
                if os.path.exists(combined_spec_path):
                    combined_spec = np.load(combined_spec_path)
                    reference_length = combined_spec.shape[-1]
                elif os.path.exists(full_spec_path):
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
            
            # Create the 5-channel spectrogram if it doesn't exist but we have all components
            if (not os.path.exists(combined_5ch_path) and 
                os.path.exists(combined_spec_path) and 
                os.path.exists(beat_frames_path)):
                    
                print(f"Creating 5-channel spectrogram for {file_name}")
                combined_spec = np.load(combined_spec_path)
                beat_frames = np.load(beat_frames_path)
                
                # Ensure beat frames match the spectrogram time dimension
                if beat_frames.shape[1] != combined_spec.shape[2]:
                    beat_frames = np.interp(
                        np.linspace(0, beat_frames.shape[1]-1, combined_spec.shape[2]),
                        np.arange(beat_frames.shape[1]),
                        beat_frames[0]
                    ).reshape(1, -1)
                    # Save the resampled beat frames
                    np.save(beat_frames_path, beat_frames)
                
                # Create 5-channel spectrogram (4 audio tracks + beat frames)
                combined_5ch = np.zeros((5, *combined_spec.shape[1:]))
                combined_5ch[:4] = combined_spec
                # Expand beat frames to match frequency dimension
                beat_expanded = np.tile(beat_frames, (n_mels, 1))
                combined_5ch[4] = beat_expanded
                np.save(combined_5ch_path, combined_5ch)

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
            "with_stems": 0,
            "with_combined": 0,
            "with_5ch": 0,
            "with_beats": 0,
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
            if base_name.endswith(('_drums', '_bass', '_vocals', '_other')):
                base_name = '_'.join(base_name.split('_')[:-1])  # Remove stem type
            base_names.add(base_name)
    
    # For each song, check what data we have
    for base_name in sorted(base_names):
        song_data = {
            "name": base_name,
            "has_full_spectrogram": False,
            "has_stems": False,
            "has_combined": False,
            "has_5ch": False,
            "has_beats": False,
            "has_brightness": False
        }
        
        # Check for each type of data
        if os.path.exists(os.path.join(output_folder, f"{base_name}_full_spectrogram.npy")):
            song_data["has_full_spectrogram"] = True
            summary["stats"]["with_full_spectrogram"] += 1
            
        stems = ["drums", "bass", "vocals", "other"]
        has_all_stems = True
        for stem in stems:
            if not os.path.exists(os.path.join(output_folder, f"{base_name}_{stem}_spectrogram.npy")):
                has_all_stems = False
                break
        
        song_data["has_stems"] = has_all_stems
        if has_all_stems:
            summary["stats"]["with_stems"] += 1
            
        if os.path.exists(os.path.join(output_folder, f"{base_name}_combined_spectrogram.npy")):
            song_data["has_combined"] = True
            summary["stats"]["with_combined"] += 1
            
        if os.path.exists(os.path.join(output_folder, f"{base_name}_combined_5ch_spectrogram.npy")):
            song_data["has_5ch"] = True
            summary["stats"]["with_5ch"] += 1
            
        if os.path.exists(os.path.join(output_folder, f"{base_name}_beatframes.npy")):
            song_data["has_beats"] = True
            summary["stats"]["with_beats"] += 1
            
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
    print(f"With all stems: {summary['stats']['with_stems']}")
    print(f"With combined spectrogram: {summary['stats']['with_combined']}")
    print(f"With 5-channel spectrogram: {summary['stats']['with_5ch']}")
    print(f"With beat frames: {summary['stats']['with_beats']}")
    print(f"With brightness data: {summary['stats']['with_brightness']}")
# Modified to include hop_length parameter
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

# Example usage
if __name__ == "__main__":
    path = Path(os.getcwd()) / "lumasync"
    preprocess_data(str(path / "data"), str(path / "preprocessed_data"), sr=22050, n_mels=128, hop_length=512)