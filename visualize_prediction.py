import os
import subprocess
import numpy as np
import cv2
import librosa
import torch
import soundfile as sf
from pathlib import Path
from demucs import pretrained
from demucs.apply import apply_model
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

def separate_and_combine_sources(file_path, output_dir):
    """
    Separates the audio into sources using HTDemucs and calculates onset functions for each source.

    Parameters:
    - file_path: Path to the input audio file.
    - output_dir: Directory to save the separated sources.

    Returns:
    - sources_onsets: Dictionary containing audio and onset functions for each source.
    - sr: Sampling rate of the audio.
    """
    # Load the HTDemucs model
    model = pretrained.get_model('htdemucs')

    # Load the audio file as a waveform
    audio, sr = librosa.load(file_path, sr=None, mono=False)  # Load stereo audio
    audio = torch.tensor(audio, dtype=torch.float32)  # Convert to PyTorch tensor
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension for mono audio
    audio = audio.unsqueeze(0)  # Add batch dimension for Demucs

    # Separate the audio into sources
    with torch.no_grad():  # Disable gradient computation for inference
        sources = apply_model(model, audio, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Check the number of sources returned by the model
    num_sources = sources.shape[1]
    sources = sources.squeeze(0)  # Remove batch dimension
    print(f"Number of sources returned by the model: {num_sources}")

    # Initialize a dictionary to store onset functions for each source
    sources_onsets = {}

    print(f"Calculating onset functions for {num_sources} sources...")
    # Process each source
    for i in range(num_sources):
        source = sources[i]
        source_mono = source.mean(dim=0).cpu().numpy()  # Convert to mono
        onset_env = librosa.onset.onset_strength_multi(y=source_mono, sr=sr)  # Calculate onset function
        sources_onsets[f"source_{i}"] = {
            "audio": source_mono,
            "onset": onset_env
        }

    print("Saving separated sources...")
    # Combine all sources into one track
    combined_audio = sum(sources).mean(dim=0).cpu().numpy()
    combined_onset = librosa.onset.onset_strength_multi(y=combined_audio, sr=sr)
    sources_onsets["combined"] = {
        "audio": combined_audio,
        "onset": combined_onset
    }

    return sources_onsets, sr

def create_brightness_representations(brightness, fps=30):
    """
    Creates different representations of brightness values with improved peak detection.
    
    Parameters:
    - brightness: Raw brightness values array
    - fps: Frames per second for time-dependent processing
    
    Returns:
    - Dictionary of different brightness representations
    """
    # Normalize brightness to 0-1 range
    brightness_norm = (brightness - np.min(brightness)) / (np.max(brightness) - np.min(brightness))

    # Make last 10 values average of track
    brightness_norm[-10:] = np.mean(brightness_norm)
    
    representations = {
        "Raw": brightness_norm,
    }
    
    # 1. Smoothed versions (different levels)
    window_small = int(fps * 0.1)  # 100ms window
    window_medium = int(fps * 0.5)  # 500ms window
    window_large = int(fps * 1.0)  # 1s window
    
    # Ensure windows are odd for savgol_filter
    window_small = window_small if window_small % 2 == 1 else window_small + 1
    window_medium = window_medium if window_medium % 2 == 1 else window_medium + 1
    window_large = window_large if window_large % 2 == 1 else window_large + 1
    
    representations["Smooth (100ms)"] = savgol_filter(brightness_norm, window_small, 2)
    representations["Smooth (500ms)"] = savgol_filter(brightness_norm, window_medium, 2)
    representations["Smooth (1s)"] = savgol_filter(brightness_norm, window_large, 2)
    
    # NEW: Create locally-centered versions that highlight peaks by removing local average
    # For raw brightness
    local_avg_window = int(fps * 2)  # 2-second window for local average
    kernel = np.ones(local_avg_window) / local_avg_window
    local_avg_raw = np.convolve(brightness_norm, kernel, mode='same')
    centered_raw = brightness_norm - local_avg_raw
    # Normalize to 0-1 range
    centered_raw = (centered_raw - np.min(centered_raw)) / (np.max(centered_raw) - np.min(centered_raw))
    representations["Raw Centered"] = centered_raw
    
    # For smoothed versions
    smooth_100ms = representations["Smooth (100ms)"]
    local_avg_100ms = np.convolve(smooth_100ms, kernel, mode='same')
    centered_100ms = smooth_100ms - local_avg_100ms
    centered_100ms = (centered_100ms - np.min(centered_100ms)) / (np.max(centered_100ms) - np.min(centered_100ms))
    representations["Smooth 100ms Centered"] = centered_100ms
    
    smooth_500ms = representations["Smooth (500ms)"]
    local_avg_500ms = np.convolve(smooth_500ms, kernel, mode='same')
    centered_500ms = smooth_500ms - local_avg_500ms
    centered_500ms = (centered_500ms - np.min(centered_500ms)) / (np.max(centered_500ms) - np.min(centered_500ms))
    representations["Smooth 500ms Centered"] = centered_500ms
    
    # 2. Derivative (rate of change) - keep existing code
    rate_of_change = np.zeros_like(brightness_norm)
    rate_of_change[1:-1] = (brightness_norm[2:] - brightness_norm[:-2]) / 2
    # Normalize to 0-1
    rate_of_change = (rate_of_change - np.min(rate_of_change)) / (np.max(rate_of_change) - np.min(rate_of_change)) if np.max(rate_of_change) > np.min(rate_of_change) else np.zeros_like(rate_of_change)
    representations["Rate of Change"] = rate_of_change
    
    # Rest of your existing code...
    # (Keep the remaining calculations for acceleration, peaks detection, etc.)
    
    # 3. Second derivative (acceleration)
    acceleration = np.zeros_like(brightness_norm)
    acceleration[1:-1] = (rate_of_change[2:] - rate_of_change[:-2]) / 2
    # Normalize to 0-1
    acceleration = (acceleration - np.min(acceleration)) / (np.max(acceleration) - np.min(acceleration)) if np.max(acceleration) > np.min(acceleration) else np.zeros_like(acceleration)
    representations["Acceleration"] = acceleration
    
    # 4. IMPROVED PEAK DETECTION - Use the 500ms smoothed signal for consistency
    smooth_signal = representations["Smooth (500ms)"]
    
    # Calculate adaptive threshold based on signal statistics
    mean_brightness = np.mean(smooth_signal)
    std_brightness = np.std(smooth_signal)
    
    # Major peaks - strong prominence, wider spacing
    major_peaks, major_properties = find_peaks(
        smooth_signal, 
        prominence=0.15,                # Higher prominence threshold for major peaks
        distance=int(fps * 0.5),        # At least 0.5 second between peaks
        height=mean_brightness + 0.3*std_brightness,  # Must be above mean + fraction of std
        width=2                         # Require some width to avoid noise spikes
    )
    
    # Medium peaks - moderate prominence
    medium_peaks, medium_properties = find_peaks(
        smooth_signal, 
        prominence=0.08,                # Medium prominence threshold
        distance=int(fps * 0.3),        # Moderate spacing requirement
        height=mean_brightness + 0.1*std_brightness  # Lower height threshold
    )
    
    # Create the peak indicators with different heights based on importance
    peak_indicator = np.zeros_like(brightness_norm)
    
    # Set medium peaks at 0.5 height
    peak_indicator[medium_peaks] = 0.5
    
    # Set major peaks at full height (will overwrite any medium peaks that are also major)
    peak_indicator[major_peaks] = 1.0
    
    # Spread peaks slightly for visibility
    peak_kernel = np.array([0.2, 0.5, 1.0, 0.5, 0.2])  # Better shaped kernel
    peak_kernel = peak_kernel / np.sum(peak_kernel)     # Normalize
    peak_indicator = np.convolve(peak_indicator, peak_kernel, mode='same')
    
    representations["Peaks"] = peak_indicator
    
    # Continue with the rest of your existing code...

    return representations

def process_and_render_video_with_audio(name):
    """
    Processes the audio and prediction data, renders a video with meter bars on a black screen,
    and combines the rendered video with the song in the background.

    Parameters:
    - name: Name of the input audio file (without extension).
    """
    # Paths for input and output
    path = Path(os.getcwd())
    visualizations_dir = path / "visualizations" / name
    audio_path = next(path.glob(f"{name}.*"), None)
    if audio_path is None:
        print(f"Error: Audio file not found for {name}.")
        return
    brightness_path = path / "predictions" / f"{name}_brightness.npy"
    temp_video_path = visualizations_dir / "temp_video.mp4"
    output_video_path = visualizations_dir / "output_video.mp4"
    separated_audio_dir = visualizations_dir / "separated_audio"

    # Create the visualizations directory
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # Move the audio file to the visualizations directory
    new_audio_path = visualizations_dir / f"{name}.wav"
    if not new_audio_path.exists():
        audio_path.rename(new_audio_path)

    # Check if required files exist
    if not new_audio_path.exists():
        print(f"Error: Audio file not found at {new_audio_path}")
        return
    if not brightness_path.exists():
        print(f"Error: Brightness data not found at {brightness_path}")
        return
    
    # Load audio and prediction data
    print("Loading audio and prediction data...")
    audio, sr = librosa.load(new_audio_path, sr=None)
    predicted_brightness = np.load(brightness_path)
    
    # Create multiple representations of brightness
    fps = 30
    brightness_representations = create_brightness_representations(predicted_brightness, fps)
    print(f"Created {len(brightness_representations)} different brightness representations")

    # Step 1: Separate and combine audio sources
    os.makedirs(separated_audio_dir, exist_ok=True)
    sources_onsets, sr = separate_and_combine_sources(new_audio_path, separated_audio_dir)
    
    print("Enhancing onset functions...")
    for key in sources_onsets.keys():
        # Ensure audio_onset is a 1D array
        if sources_onsets[key]["onset"].ndim > 1:
            sources_onsets[key]["onset"] = sources_onsets[key]["onset"].mean(axis=0)

    # Define labels for the sources
    source_labels = {
        "source_0": "Drums",
        "source_1": "Bass",
        "source_2": "Other",
        "source_3": "Vocals",
        "combined": "Combined"
    }

    # Use the combined audio if no instrument is specified
    instrument_audio_path = os.path.join(separated_audio_dir, "combined_track.wav")
    sf.write(instrument_audio_path, sources_onsets["combined"]["audio"], sr)
    print(f"Combined audio saved to {instrument_audio_path}")

    # Step 2: Initialize a 1080p black video
    width, height = 1920, 1080  # 1080p resolution
    fps = 30
    duration = len(audio) / sr
    total_frames = int(duration * fps)

    # Synchronize onsets with video frames
    print("Synchronizing onsets with video frames...")
    for key in sources_onsets.keys():
        onset_length = len(sources_onsets[key]["onset"])
        if onset_length != total_frames:
            sources_onsets[key]["onset"] = np.interp(
                np.linspace(0, onset_length - 1, total_frames),
                np.arange(onset_length),
                sources_onsets[key]["onset"]
            )
            
    # Synchronize all brightness representations
    for key in brightness_representations:
        rep_length = len(brightness_representations[key])
        if rep_length != total_frames:
            brightness_representations[key] = np.interp(
                np.linspace(0, rep_length - 1, total_frames),
                np.arange(rep_length),
                brightness_representations[key]
            )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    max_height = 500  # Maximum height of the bars in pixels
    bar_width = 30
    bar_spacing = 60
    
    # Define colors for different representations (in BGR format for OpenCV)
    colors = {
        "Raw": (255, 255, 0),              # Yellow
        "Raw Centered": (200, 200, 0),     # Dark yellow
        "Smooth (100ms)": (0, 255, 0),     # Green
        "Smooth 100ms Centered": (0, 200, 0),  # Dark green
        "Smooth (500ms)": (0, 255, 255),   # Yellow-green
        "Smooth 500ms Centered": (0, 200, 200),  # Dark yellow-green
        "Smooth (1s)": (0, 165, 255),      # Orange
        "Rate of Change": (255, 0, 0),     # Blue
        "Acceleration": (128, 0, 128),     # Purple
        "Peaks": (0, 0, 255),              # Red
        "Thresholded": (255, 255, 255),    # White
        "MovAvg (1s)": (255, 0, 255),      # Pink
        "MovAvg (2s)": (200, 0, 200),      # Dark pink
        "MovAvg (4s)": (150, 0, 150),      # Darker pink
        "Peak Density": (0, 128, 255),     # Orange
        "Onset Strength": (0, 255, 128),   # Teal
    }

    print("Rendering video with meters...")
    for frame_index in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Render bars for each source's onset function
        for i, (key, value) in enumerate(sources_onsets.items()):
            if frame_index < len(value["onset"]):
                onset_strength = value["onset"][frame_index]
                max_onset = np.max(value["onset"]) if len(value["onset"]) > 0 else 1  # Avoid division by zero
                bar_height = int((onset_strength / max_onset) * max_height)
                bar_x = 50 + i * bar_spacing
                bar_y = height // 2 - bar_height
                color = (0, 255, 0)
                label = source_labels.get(key, key)  # Use the label from the mapping or default to the key
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, height // 2), color, -1)
                cv2.putText(frame, label, (bar_x, height // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add separator line
        cv2.line(frame, (500, height//2 - max_height - 20), (500, height//2 + 20), (50, 50, 50), 2)

        brightness_start_x = 580
        for i, (key, values) in enumerate(brightness_representations.items()):
            if frame_index < len(values):
                # Get the raw value
                raw_value = values[frame_index]
                enhanced_value = raw_value
                
                # Apply more aggressive non-linear scaling for greater visual contrast
                if "Rate of Change" in key or "Acceleration" in key:
                    # Make sure the value is non-negative before applying power function
                    abs_value = np.abs(raw_value)  # Use absolute value for rate of change/acceleration
                    enhanced_value = np.power(abs_value, 3)
                    # Preserve the sign to show direction of change
                    if raw_value < 0:
                        enhanced_value = -enhanced_value
                elif "Centered" in key:
                    # For centered signals, apply less aggressive scaling
                    enhanced_value = np.power(np.maximum(0, raw_value), 1.5)  # Lighter power scaling
                
                # Handle any remaining NaN values (though this shouldn't happen now)
                if np.isnan(enhanced_value):
                    enhanced_value = 0
                
                # Calculate bar height with enhanced contrast
                bar_height = int(enhanced_value * max_height)
                
                # Ensure very small values are at least 1 pixel for visibility
                if enhanced_value > 0 and bar_height == 0:
                    bar_height = 1
                
                bar_x = brightness_start_x + i * bar_spacing
                bar_y = height // 2 - bar_height
                color = colors.get(key, (255, 255, 255))  # Default to white if no color defined
                
                # Draw the representation bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, height // 2), color, -1)
                
                # Add markers at 25%, 50% and 75% of original value for reference
                if key == "Raw":
                    # Draw lines at 25%, 50%, and 75% of the raw value range
                    for percent in [0.25, 0.5, 0.75]:
                        marker_y = height // 2 - int(percent * max_height)
                        cv2.line(frame, (bar_x-5, marker_y), (bar_x+bar_width+5, marker_y), 
                                (100, 100, 100), 1)
                
                # Write label vertically to save space
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                # Calculate the position to center text below the bar
                text_x = bar_x + (bar_width - text_size[1]) // 2  # Center text below bar
                text_y = height // 2 + 50
                
                # Rotate and position the text
                for j, char in enumerate(key):
                    char_y = text_y + j * 10  # Space characters vertically
                    cv2.putText(frame, char, (text_x, char_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                # Show the actual value percentage above the bar for raw brightness
                if key == "Raw":
                    value_text = f"{int(raw_value * 100)}%"
                    cv2.putText(frame, value_text, (bar_x, height // 2 - max_height - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Write the frame to the video
        out.write(frame)

    out.release()
    print(f"Rendered video saved to {temp_video_path}")

    # Steps 3 and 4 unchanged...
    # [rest of the function]

    # Step 3: Combine the rendered video with the selected instrument's audio
    command = [
        'ffmpeg',
        '-y',
        '-i', str(temp_video_path),
        '-i', str(instrument_audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        str(output_video_path)
    ]
    subprocess.run(command, check=True)
    print(f"Final video with audio saved to {output_video_path}")

    # Clean up temporary video file
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    print(f"Temporary video file {temp_video_path} removed.")

def detect_brightness_peaks(brightness, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.2, wait=5):
    """
    Detect peaks in the brightness values using librosa's peak_pick function.
    """
    peaks = librosa.util.peak_pick(
        x=brightness,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait
    )
    return peaks

process_and_render_video_with_audio('californialove')