from pathlib import Path
from preprocess import preprocess_data
from train import train_model
import os

def start_pipeline(input_folder, 
                    output_folder, 
                    epochs=10, 
                    batch_size=8, 
                    lr=1e-4, 
                    sr=22050, 
                    n_mels=128, 
                    resume_from_best=False,
                    transformer_dim=512,
                    channels=["full"],
                    max_time_dim=2048,
                    transformer_chunk_size=2048):
    """
    Combines preprocessing and training into a single pipeline.

    Args:
        input_folder (str): Path to the folder containing `.mov` files.
        output_folder (str): Path to save the preprocessed data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for training.
        sr (int): Sampling rate for the audio.
        n_mels (int): Number of Mel bands.
        resume_from_best (bool): Whether to resume training from the best checkpoint.
        transformer_dim (int): Dimension of the transformer model.
        channels (list): List of channels to use for training.
    """

    # App path
    app_path = "lumasync"

    # Get the current working directory
    current_path = Path(os.getcwd())

    # Resolve input and output folder paths
    input_folder = current_path / app_path / input_folder
    output_folder = current_path / app_path / output_folder

    # Ensure input folder exists
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: Preprocessing data from {input_folder}...")
    preprocess_data(str(input_folder), str(output_folder), sr=sr, n_mels=n_mels)

    print(f"Step 2: Training the model with data from {output_folder}...")
    train_model(str(output_folder), 
                epochs=epochs, 
                batch_size=batch_size, 
                lr=lr, 
                resume_from_best=resume_from_best,
                transformer_dim=transformer_dim,
                channels=channels,
                max_time_dim=max_time_dim,
                transformer_chunk_size=transformer_chunk_size)

    print("Pipeline completed successfully!")

# Example usage
if __name__ == "__main__":
    input_folder = "data"  # Folder containing `.mov` files (relative to current directory)
    output_folder = "preprocessed_data"  # Folder to save preprocessed data (relative to current directory)
    start_pipeline(input_folder, 
                   output_folder, 
                   epochs=10, 
                   batch_size=8, 
                   lr=1e-4, 
                   resume_from_best=False,
                    transformer_dim=1024,
                    channels=["drums", "bass", "other", "vocals"],
                    max_time_dim=None,
                    transformer_chunk_size=None)