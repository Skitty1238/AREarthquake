import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------
# Function Definitions
# ---------------------

def load_waveform_image(image_path):
    """
    Loads an image from disk and converts it to a grayscale NumPy array.
    """
    image = Image.open(image_path).convert('L')
    return np.array(image)

def extract_waveform_from_image(image_data):
    """
    Extracts a 1D amplitude curve by finding the bright waveform's vertical position relative to the center.
    For each column in the image, bright pixels (above 50% of that column's max brightness) are averaged;
    this average is then converted into an amplitude where the image's vertical center is 0.
    """
    height, width = image_data.shape
    mid = height / 2.0  # vertical center
    amplitude_curve = np.zeros(width, dtype=np.float32)
    
    for col in range(width):
        column_pixels = image_data[:, col]
        col_max = np.max(column_pixels)
        threshold = col_max * 0.5  # adjust this factor as needed
        
        # Indices where the pixel brightness exceeds the threshold
        indices = np.where(column_pixels >= threshold)[0]
        if len(indices) > 0:
            waveform_index = np.mean(indices)
        else:
            waveform_index = mid  # default to center if no bright pixels
        
        # Map vertical distance from the center to an amplitude (-1 to 1)
        amplitude = (mid - waveform_index) / mid
        amplitude_curve[col] = amplitude

    return amplitude_curve

def load_audio(audio_path):
    """
    Loads audio from a file using librosa (preserves original sample rate).
    """
    audio, sr = librosa.load(audio_path, sr=None)
    return audio, sr

def resample_curve_to_audio_length(amplitude_curve, audio_length):
    """
    Resamples the amplitude curve (1D array) to match the length of the audio signal.
    """
    img_length = len(amplitude_curve)
    if img_length == audio_length:
        return amplitude_curve
    
    x_original = np.linspace(0, 1, img_length)
    x_target = np.linspace(0, 1, audio_length)
    resampled_curve = np.interp(x_target, x_original, amplitude_curve)
    return resampled_curve

def apply_amplitude_envelope(audio, envelope):
    """
    Multiplies the audio signal sample-by-sample by the envelope.
    """
    if len(audio) != len(envelope):
        raise ValueError("Audio and envelope lengths must match.")
    return audio * envelope

def save_audio(audio, sr, output_path):
    """
    Writes the adjusted audio to a WAV file.
    """
    sf.write(output_path, audio, sr)

def save_plot_results(image_data, raw_amplitude_curve, envelope, original_audio, adjusted_audio, plot_output_path):
    """
    Generates a multi-panel plot showing:
      1. The original waveform image.
      2. The raw extracted amplitude curve (before resampling).
      3. The resampled envelope matching the audio length.
      4. A comparison of the original and adjusted audio waveforms.
    The plot is then saved to the provided path.
    """
    plt.figure(figsize=(12, 10))
    
    # Plot the original image
    plt.subplot(4, 1, 1)
    plt.imshow(image_data, cmap='gray', aspect='auto')
    plt.title("Original Waveform Image")
    
    # Plot the raw amplitude curve (from image columns)
    plt.subplot(4, 1, 2)
    plt.plot(raw_amplitude_curve, color='blue')
    plt.title("Raw Extracted Amplitude Curve (Before Resampling)")
    
    # Plot the envelope after resampling
    plt.subplot(4, 1, 3)
    plt.plot(envelope, color='green')
    plt.title("Resampled Envelope (Matches Audio Length)")
    
    # Plot the audio waveforms (original and adjusted)
    plt.subplot(4, 1, 4)
    plt.plot(original_audio, label='Original Audio', alpha=0.5)
    plt.plot(adjusted_audio, label='Adjusted Audio', alpha=0.8)
    plt.title("Audio Waveform Comparison")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()

def process_image_and_save_audio(image_path, audio, sr, output_dir, plot_dir=None):
    """
    Processes a single image: extracts its amplitude envelope, resamples to match
    the audio, applies the envelope to the audio, and saves both the adjusted audio
    and a corresponding plot. The output filenames are based on the image's base name.
    """
    # Load and process the image
    image_data = load_waveform_image(image_path)
    raw_amplitude_curve = extract_waveform_from_image(image_data)
    envelope = resample_curve_to_audio_length(raw_amplitude_curve, len(audio))
    
    # Apply the envelope to the audio
    adjusted_audio = apply_amplitude_envelope(audio, envelope)
    
    # Construct the output audio filename: original image name + "-audio.wav"
    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    output_filename = f"{name}-audio.wav"
    output_audio_path = os.path.join(output_dir, output_filename)
    
    # Save the adjusted audio
    save_audio(adjusted_audio, sr, output_audio_path)
    print(f"Saved adjusted audio for '{base}' as '{output_filename}'.")
    
    # Save the plot if a plot directory is provided
    if plot_dir is not None:
        plot_filename = f"{name}-plot.png"
        plot_output_path = os.path.join(plot_dir, plot_filename)
        save_plot_results(image_data, raw_amplitude_curve, envelope, audio, adjusted_audio, plot_output_path)
        print(f"Saved plot for '{base}' as '{plot_filename}'.")

# ---------------------
# Main Processing Loop
# ---------------------

if __name__ == "__main__":
    # Define directories for images, audio output, and plots
    image_dir = "/Users/aaravkumar/Desktop/seismo-cropped"
    output_dir = "/Users/aaravkumar/Desktop/seismo-audio"
    plot_dir = os.path.join(output_dir, "plots")
    
    # Path to the audio file that will be processed with each image
    audio_path = "/Users/aaravkumar/Desktop/eq-shortened.wav"
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load the audio file only once
    audio, sr = load_audio(audio_path)
    
    # Process every valid image file in the given directory
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(valid_extensions):
            full_image_path = os.path.join(image_dir, filename)
            process_image_and_save_audio(full_image_path, audio, sr, output_dir, plot_dir)
    
    print("Batch processing complete.")