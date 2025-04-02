from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from model import LightingModel
import numpy as np
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

class LightingDataset(Dataset):
    def __init__(self, data_folder, use_5ch=False):  # Default changed to False
        self.data_folder = data_folder
        self.use_5ch = use_5ch
        
        # Look for 4-channel spectrograms by default
        self.files = [f for f in os.listdir(data_folder) if f.endswith("_combined_spectrogram.npy")]
        
        # Only check for 5-channel if explicitly requested
        if use_5ch and not self.files:
            self.files = [f for f in os.listdir(data_folder) if f.endswith("_combined_5ch_spectrogram.npy")]
        
        print(f"Found {len(self.files)} spectrogram files for training")
        
        # Generate visualizations for a few samples
        if len(self.files) > 0:
            self.generate_sample_visualizations(3)  # Visualize first 3 files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load spectrograms - always use only 4 channels regardless of source
        spectrogram_path = os.path.join(self.data_folder, self.files[idx])
        
        if self.files[idx].endswith("_combined_5ch_spectrogram.npy"):
            base_name = self.files[idx].replace("_combined_5ch_spectrogram.npy", "")
            # Load 5-channel but only use first 4 channels
            full_spectrogram = np.load(spectrogram_path)
            spectrogram = full_spectrogram[:4]  # Only take the first 4 channels
        else:
            base_name = self.files[idx].replace("_combined_spectrogram.npy", "")
            # Load 4-channel directly
            spectrogram = np.load(spectrogram_path)
            
        brightness_path = os.path.join(self.data_folder, f"{base_name}_brightness.npy")
        
        # Normalize each channel independently (only 4 channels)
        for i in range(spectrogram.shape[0]):
            channel = spectrogram[i]
            spectrogram[i] = (channel - channel.mean()) / (channel.std() + 1e-8)
            
        brightness = np.load(brightness_path)

        # Convert to PyTorch tensors
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        brightness = torch.tensor(brightness, dtype=torch.float32)

        return spectrogram, brightness
    
    def generate_sample_visualizations(self, num_samples=3):
        """
        Generate visualizations showing the relationship between drums and brightness.
        """
        os.makedirs("visualizations", exist_ok=True)
        
        # Get a subset of files to visualize
        vis_files = self.files[:min(num_samples, len(self.files))]
        
        for file_name in vis_files:
            if file_name.endswith("_combined_5ch_spectrogram.npy"):
                base_name = file_name.replace("_combined_5ch_spectrogram.npy", "")
                spec_path = os.path.join(self.data_folder, file_name)
                spectrogram = np.load(spec_path)
                # Extract drum spectrogram (channel 0)
                drums_spec = spectrogram[0]
            else:
                base_name = file_name.replace("_combined_spectrogram.npy", "")
                spec_path = os.path.join(self.data_folder, file_name)
                spectrogram = np.load(spec_path)
                drums_spec = spectrogram[0]
            
            # Load brightness
            brightness_path = os.path.join(self.data_folder, f"{base_name}_brightness.npy")
            brightness = np.load(brightness_path)
            
            # Create visualization without beat frames
            self._create_drums_brightness_visualization(
                drums_spec, brightness, base_name
            )
    
    def _create_drums_brightness_visualization(self, drums_spec, brightness, base_name):
        """
        Create and save a visualization showing drums spectrogram with brightness.
        """
        # Set a consistent style
        plt.style.use('dark_background')
        
        # Calculate total frames and determine a good window size (6 seconds)
        total_frames = drums_spec.shape[1]
        window_size = min(200, total_frames // 3)  # ~6 sec window at 33fps
        
        # Find a section (preferably around the middle)
        mid_point = total_frames // 3
        start_frame = max(0, mid_point - window_size // 2)
        end_frame = min(total_frames, start_frame + window_size)
        
        # Create a custom colormap for drums
        colors = [(0, 0, 0), (0, 0.5, 0), (0, 1, 0), (1, 1, 0)]
        drum_cmap = LinearSegmentedColormap.from_list('DrumMap', colors)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot drum spectrogram (flipped to have low frequencies at bottom)
        drum_img = ax1.imshow(
            drums_spec[:, start_frame:end_frame], 
            aspect='auto', 
            origin='lower', 
            cmap=drum_cmap,
            extent=[start_frame, end_frame, 0, drums_spec.shape[0]]
        )
        ax1.set_title("Drum Spectrogram", color='white', fontsize=14)
        ax1.set_ylabel("Frequency", color='white')
        ax1.tick_params(colors='white')
        
        # Plot brightness on the second subplot
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
        
        # Add colorbar for spectrogram
        cbar = fig.colorbar(drum_img, ax=ax1)
        cbar.ax.tick_params(colors='white')
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(f"visualizations/{base_name}_drums_brightness.png", dpi=150)
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
    # Now expecting shape [5, freq, time] for each spectrogram (5 channels)
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

def train_model(data_folder, epochs=10, batch_size=8, lr=1e-4):
    # Load dataset with 4-channel spectrograms only (no beat frames)
    dataset = LightingDataset(data_folder, use_5ch=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model with 4 input channels (spectrograms only)
    model = LightingModel(transformer_dim=1024, in_channels=4)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Rest of the training function remains unchanged
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Create directories for saving models and checkpoints
    output_path = Path(os.getcwd()) / "lumasync"
    checkpoint_path = output_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Keep track of the best model
    best_loss = float('inf')
    
    # Training loop 
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (spectrograms, brightness, masks) in enumerate(dataloader):
            spectrograms = spectrograms.to("cuda" if torch.cuda.is_available() else "cpu")
            brightness = brightness.to("cuda" if torch.cuda.is_available() else "cpu")
            masks = masks.to("cuda" if torch.cuda.is_available() else "cpu")

            # Forward pass
            outputs = model(spectrograms)
            
            # Compatibility with both return types
            if isinstance(outputs, dict):
                outputs = outputs['combined']
            
            outputs = outputs.squeeze(-1)
            outputs = outputs * masks
            brightness = brightness * masks
            loss = criterion(outputs, brightness)

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
        
        # Save checkpoint every 3 epochs
        if (epoch + 1) % 3 == 0:
            checkpoint_file = checkpoint_path / f"model_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_file)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_file}")
        
        # Save the best model based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = output_path / "best_model_no_beats.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {avg_loss:.6f}")

    # Save the final trained model
    final_model_path = output_path / "trained_model_no_beats.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as '{final_model_path}'")
    
    # Also save a complete checkpoint with training state
    final_checkpoint_path = checkpoint_path / "final_checkpoint.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
    }, final_checkpoint_path)
    print(f"Final checkpoint saved at '{final_checkpoint_path}'")

    return final_model_path, best_model_path

# Example usage
if __name__ == "__main__":
    train_model("preprocessed_data", epochs=20, batch_size=4)