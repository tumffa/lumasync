from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import LightingModel
import numpy as np
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

class LightingDataset(Dataset):
    def __init__(self, data_folder, channels=["full"]):
        """
        Args:
            data_folder: Directory containing preprocessed data
            channels: List of channels to use (e.g. ["full"] or ["drums", "bass"])
        """
        self.data_folder = data_folder
        self.channels = channels
        
        print(f"Looking for songs with channels: {', '.join(channels)}")
        
        # Find all available spectrograms grouped by song/base name
        self.song_specs = self.find_available_spectrograms(data_folder)
        
        # Print stats about available data
        self.print_dataset_stats()
    
    def find_available_spectrograms(self, data_folder):
        """Find all available spectrograms and organize them by song."""
        all_files = os.listdir(data_folder)
        
        # Map to store songs and their available spectrograms
        song_specs = {}
        
        # Process all spectrogram files
        for file in all_files:
            # Check if this is a spectrogram file for any supported channel
            for channel in ["full", "drums", "bass", "vocals", "other", "drums2"]:
                if file.endswith(f"_{channel}_spectrogram.npy"):
                    # Extract base name (everything before the channel marker)
                    base_name = file.replace(f"_{channel}_spectrogram.npy", "")
                    
                    # Initialize entry if new song
                    if base_name not in song_specs:
                        song_specs[base_name] = {
                            "channels": [],
                            "has_brightness": False
                        }
                    
                    # Add this channel
                    song_specs[base_name]["channels"].append(channel)
                    break  # Found the channel, no need to check others
                    
        # Check which songs have brightness data
        for base_name in list(song_specs.keys()):
            brightness_path = os.path.join(data_folder, f"{base_name}_brightness.npy")
            if os.path.exists(brightness_path):
                song_specs[base_name]["has_brightness"] = True
            else:
                # Remove songs without brightness data
                del song_specs[base_name]
        
        # Filter songs based on requested channels
        valid_songs = {}
        for base_name, data in song_specs.items():
            # Check if all requested channels are available
            if all(channel in data["channels"] for channel in self.channels):
                valid_songs[base_name] = data
        
        return valid_songs
    
    def print_dataset_stats(self):
        """Print statistics about the dataset."""
        total_songs = len(self.song_specs)
        
        # Count songs with each channel type
        channel_counts = {channel: 0 for channel in ["full", "drums", "bass", "vocals", "other", "drums2"]}
        for data in self.song_specs.values():
            for channel in data["channels"]:
                channel_counts[channel] += 1
        
        print(f"Dataset summary:")
        print(f"Total songs with spectrograms and brightness data: {total_songs}")
        for channel, count in channel_counts.items():
            if count > 0:
                print(f"  - Songs with {channel} channel: {count}")
        
        # Calculate how many songs match our requested channels
        valid_songs = len([base_name for base_name, data in self.song_specs.items() 
                         if all(ch in data["channels"] for ch in self.channels)])
        
        print(f"Songs matching requested channels ({', '.join(self.channels)}): {valid_songs}")
        
        if valid_songs == 0:
            print("\nWARNING: No songs found with all requested channels!")
            print("Available channel combinations:")
            
            # Find most common channel combinations
            combinations = {}
            for data in self.song_specs.values():
                channels_tuple = tuple(sorted(data["channels"]))
                combinations[channels_tuple] = combinations.get(channels_tuple, 0) + 1
            
            # Print available combinations
            for channels, count in sorted(combinations.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {', '.join(channels)}: {count} songs")
        else:
            # Generate sample visualizations
            sample_songs = list(self.song_specs.keys())[:min(3, len(self.song_specs))]
            for song in sample_songs:
                self.generate_visualization(song)
    
    def __len__(self):
        """Return the number of valid songs in the dataset."""
        return len([base_name for base_name, data in self.song_specs.items() 
                  if all(ch in data["channels"] for ch in self.channels)])
    
    def __getitem__(self, idx):
        """Load the data for the song at the given index."""
        # Get the base name for the song at this index
        valid_songs = [base_name for base_name, data in self.song_specs.items() 
                     if all(ch in data["channels"] for ch in self.channels)]
        
        if idx >= len(valid_songs):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(valid_songs)} items")
        
        base_name = valid_songs[idx]
        
        # Load spectrograms for requested channels
        spectrograms = []
        for channel in self.channels:
            spec_path = os.path.join(self.data_folder, f"{base_name}_{channel}_spectrogram.npy")
            spec = np.load(spec_path)
            
            # Ensure 2D shape
            if spec.ndim == 3:
                spec = spec[0]  # Take the first channel if it's 3D
            
            # Normalize
            mean = np.mean(spec)
            std = np.std(spec) + 1e-8
            spec = (spec - mean) / std
            
            spectrograms.append(spec)
        
        # Stack channels
        multi_channel_spec = np.stack(spectrograms)
        
        # Load brightness
        brightness_path = os.path.join(self.data_folder, f"{base_name}_brightness.npy")
        brightness = np.load(brightness_path)
        
        # Convert to PyTorch tensors
        spectrogram_tensor = torch.tensor(multi_channel_spec, dtype=torch.float32)
        brightness_tensor = torch.tensor(brightness, dtype=torch.float32)
        
        return spectrogram_tensor, brightness_tensor
    
    def generate_visualization(self, base_name):
        """Generate a visualization for a specific song."""
        # Create output directory
        os.makedirs("visualizations", exist_ok=True)
        
        # Choose first available channel from requested channels for visualization
        channel = None
        for ch in self.channels:
            if ch in self.song_specs[base_name]["channels"]:
                channel = ch
                break
        
        if not channel:
            return  # No matching channel
            
        # Load spectrogram and brightness
        spec_path = os.path.join(self.data_folder, f"{base_name}_{channel}_spectrogram.npy")
        brightness_path = os.path.join(self.data_folder, f"{base_name}_brightness.npy")
        
        spec = np.load(spec_path)
        brightness = np.load(brightness_path)
        
        # Ensure 2D for visualization
        if spec.ndim == 3:
            spec = spec[0]
            
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(spec, aspect='auto', origin='lower')
        plt.title(f"{base_name} - {channel} spectrogram")
        plt.colorbar()
        
        plt.subplot(2, 1, 2)
        plt.plot(brightness)
        plt.title("Brightness")
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"visualizations/{base_name}_{channel}_vis.png")
        plt.close()
        
        print(f"Generated visualization for {base_name}")
    
    def generate_sample_visualizations(self, num_samples=3):
        """Generate visualizations showing spectrograms and brightness."""
        os.makedirs("visualizations", exist_ok=True)
        
        # Get a subset of files to visualize
        vis_files = self.base_files[:min(num_samples, len(self.base_files))]
        
        for base_name in vis_files:
            # For visualization, pick the first channel in our list
            channel_to_vis = self.channels[0]
            
            if channel_to_vis == "full":
                spec_path = os.path.join(self.data_folder, f"{base_name}_full_spectrogram.npy")
            else:
                spec_path = os.path.join(self.data_folder, f"{base_name}_{channel_to_vis}_spectrogram.npy")
                
            spec = np.load(spec_path)
            # Ensure 2D for visualization
            if spec.ndim == 3:
                spec = spec[0]
            
            # Load brightness
            brightness_path = os.path.join(self.data_folder, f"{base_name}_brightness.npy")
            brightness = np.load(brightness_path)
            
            # Create visualization
            self._create_spectrogram_visualization(
                spec, brightness, base_name, channel_to_vis
            )
    
    def _create_spectrogram_visualization(self, spectrogram, brightness, base_name, channel_name):
        """Create visualization showing spectrogram with brightness."""
        plt.style.use('dark_background')
        
        # Calculate window size
        total_frames = spectrogram.shape[1]
        window_size = min(200, total_frames // 3)
        
        # Find a good section
        mid_point = total_frames // 3
        start_frame = max(0, mid_point - window_size // 2)
        end_frame = min(total_frames, start_frame + window_size)
        
        # Create a custom colormap based on channel
        if channel_name == "drums":
            colors = [(0, 0, 0), (0, 0.5, 0), (0, 1, 0), (1, 1, 0)]
        elif channel_name == "bass":
            colors = [(0, 0, 0), (0.5, 0, 0), (1, 0, 0), (1, 1, 0)]
        elif channel_name == "vocals":
            colors = [(0, 0, 0), (0, 0, 0.5), (0, 0, 1), (1, 1, 1)]
        else:  # other or combined
            colors = [(0, 0, 0), (0, 0.5, 0.7), (0, 0.8, 1), (1, 1, 1)]
            
        spec_cmap = LinearSegmentedColormap.from_list('SpecMap', colors)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot spectrogram
        spec_img = ax1.imshow(
            spectrogram[:, start_frame:end_frame], 
            aspect='auto', 
            origin='lower', 
            cmap=spec_cmap,
            extent=[start_frame, end_frame, 0, spectrogram.shape[0]]
        )
        ax1.set_title(f"{channel_name.capitalize()} Spectrogram", color='white', fontsize=14)
        ax1.set_ylabel("Frequency", color='white')
        ax1.tick_params(colors='white')
        
        # Plot brightness
        time_axis = np.arange(start_frame, end_frame)
        ax2.plot(time_axis, brightness[start_frame:end_frame], color='cyan', linewidth=2, label='Brightness')
        
        # Set titles and labels
        ax2.set_title("Brightness Values", color='white', fontsize=14)
        ax2.set_xlabel("Frame", color='white')
        ax2.set_ylabel("Brightness", color='white')
        ax2.tick_params(colors='white')
        ax2.set_ylim(0, 1)
        ax2.grid(True, color='gray', alpha=0.3)
        ax2.legend()
        
        # Add colorbar
        cbar = fig.colorbar(spec_img, ax=ax1)
        cbar.ax.tick_params(colors='white')
        
        # Save
        plt.tight_layout()
        plt.savefig(f"visualizations/{base_name}_{channel_name}_brightness.png", dpi=150)
        plt.close()
        print(f"Generated visualization for {base_name}")


def collate_fn(batch):
    """
    Custom collate function to handle variable-length spectrograms.

    Args:
        batch (list): List of (spectrogram, brightness) tuples.

    Returns:
        tuple: Padded spectrograms, brightness values, masks.
    """
    spectrograms, brightnesses = zip(*batch)

    # Find the maximum time dimension in the batch
    max_length = max(s.shape[-1] for s in spectrograms)

    # Pad spectrograms to the maximum length
    padded_spectrograms = torch.zeros(len(spectrograms), spectrograms[0].shape[0], 
                                     spectrograms[0].shape[1], max_length)
    masks = torch.zeros(len(spectrograms), max_length, dtype=torch.bool)
    
    for i, s in enumerate(spectrograms):
        padded_spectrograms[i, :, :, :s.shape[-1]] = s
        masks[i, :s.shape[-1]] = 1  # Mark valid positions as 1

    # Pad brightness values to the maximum length
    padded_brightnesses = torch.zeros(len(brightnesses), max_length)
    for i, b in enumerate(brightnesses):
        padded_brightnesses[i, :len(b)] = b

    return padded_spectrograms, padded_brightnesses, masks

def train_model(data_folder, epochs=10, batch_size=8, lr=1e-4, resume_from_best=False,
               channels=["full"], transformer_dim=512, max_time_dim=None, transformer_chunk_size=None):
    """
    Train the lighting model.
    
    Args:
        data_folder: Path to preprocessed data
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        resume_from_best: Whether to resume from the best saved model
        channels: List of audio channels to use. Options:
                 - ["full"] for single-channel approach
                 - Any subset of ["drums", "bass", "vocals", "other"] for specific stems
                 - Mix of "full" and stems (e.g. ["full", "drums"])
        transformer_dim: Dimension for transformer model
    """
    # Load dataset with specified channels
    dataset = LightingDataset(data_folder, channels=channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model with appropriate number of input channels
    in_channels = len(channels)
    model = LightingModel(transformer_dim=transformer_dim, in_channels=in_channels,
                          max_time_dim=max_time_dim, transformer_chunk_size=transformer_chunk_size)
    
    print(f"Training model with {in_channels} input channels and transformer_dim={transformer_dim}")
    
    # Create directory for saving models
    output_path = Path(os.getcwd())
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a model name based on channels
    model_name = f"model_{'_'.join(channels)}"
    
    # Keep track of the best model
    best_loss = float('inf')
    
    # If resuming from best model, load its state if it exists
    if resume_from_best:
        best_model_path = output_path / f"best_{model_name}.pth"
        if best_model_path.exists():
            print(f"Resuming training from best model at {best_model_path}")
            try:
                checkpoint = torch.load(best_model_path)
                # Handle both formats: just state_dict or dict with 'model_state_dict'
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # Check if there's stored loss
                    if 'loss' in checkpoint:
                        best_loss = checkpoint['loss']
                        print(f"Loaded previous best loss: {best_loss:.6f}")
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Warning: Could not fully load previous state: {e}")
                print("Starting with freshly initialized model weights")
        else:
            print(f"No best model found at {best_model_path}. Starting with fresh weights.")
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop 
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (spectrograms, brightness, masks) in enumerate(dataloader):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                spectrograms = spectrograms.to(device)
                brightness = brightness.to(device)
                masks = masks.to(device)

                # Store original sequence length for later
                original_seq_length = spectrograms.shape[-1]
                
                # Forward pass
                outputs = model(spectrograms)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    outputs = outputs['combined']
                
                outputs = outputs.squeeze(-1)
                
                # Resize outputs or mask if their lengths don't match
                if outputs.shape[1] != brightness.shape[1]:
                    print(f"Resizing outputs from {outputs.shape[1]} to {brightness.shape[1]} time steps")
                    # Upsample model outputs to match original sequence length
                    outputs = nn.functional.interpolate(
                        outputs.unsqueeze(1),  # [B, 1, T_smaller]
                        size=original_seq_length,
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)  # Back to [B, T_original]
                
                # Apply mask
                outputs = outputs * masks
                brightness = brightness * masks
                
                # Calculate loss
                loss = criterion(outputs, brightness)

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue  # Skip problematic batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}", end='\r')

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        
        # Step the scheduler
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save the best model based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = output_path / f"best_{model_name}_dim_{transformer_dim}_b{batch_size}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'channels': channels,
                'transformer_dim': transformer_dim,
            }, best_model_path)
            print(f"New best model saved with loss: {avg_loss:.6f}")

    # Save the final trained model
    final_model_path = output_path / f"trained_{model_name}_dim_{transformer_dim}_b{batch_size}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': avg_loss,
        'channels': channels,
        'transformer_dim': transformer_dim,
    }, final_model_path)
    print(f"Final model saved as '{final_model_path}'")

    return final_model_path, best_model_path

# Example usage
if __name__ == "__main__":
    # Train with all 4 separated channels (default)
    train_model("preprocessed_data", epochs=10, batch_size=4, resume_from_best=False,
               channels=["full"])
