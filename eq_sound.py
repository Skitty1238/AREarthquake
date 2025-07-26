import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load the waveform image
def load_waveform_image(image_path):
    """
    Loads an image from disk and converts it to a grayscale NumPy array.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_data = np.array(image)
    return image_data

# Step 2: Extract a 1D amplitude curve from the image, using the vertical position relative to the center.
def extract_waveform_from_image(image_data):
    """
    For each column in the image, this function finds the position of the bright waveform relative to the middle.
    It then computes an amplitude value such that:
       - The vertical center (middle of the image) corresponds to zero amplitude.
       - Positions above the center yield positive values.
       - Positions below the center yield negative values.
    The amplitude is normalized by half the image height (the distance from the center to the top or bottom).
    
    Assumes the waveform is drawn in a bright color against a darker background.
    """
    height, width = image_data.shape
    mid = height / 2.0  # vertical center (as a float)
    amplitude_curve = np.zeros(width, dtype=np.float32)

    # Process each column individually
    for col in range(width):
        column_pixels = image_data[:, col]

        # Compute a threshold based on the maximum brightness in the column.
        # This helps in dealing with thick waveforms (by only considering the "bright" part).
        col_max = np.max(column_pixels)
        threshold = col_max * 0.5  # you can adjust this threshold factor as needed

        # Find the indices where the pixel brightness is above the threshold.
        indices = np.where(column_pixels >= threshold)[0]

        if len(indices) > 0:
            # Use the average row index of the bright pixels.
            waveform_index = np.mean(indices)
        else:
            # If no bright pixel is detected in this column, assume the waveform is at the center.
            waveform_index = mid

        # Map the vertical distance from the center to an amplitude value.
        # Here, if the waveform is at the top (index=0), then (mid - 0) / mid = +1.
        # If at the bottom (index=height-1), then (mid - (height-1)) / mid will be negative.
        amplitude = (mid - waveform_index) / mid
        amplitude_curve[col] = amplitude

    # At this point, amplitude_curve is defined per column.
    # Typical values range from approximately -1 to +1 depending on the waveform's vertical position.
    return amplitude_curve

# Step 3: Load audio
def load_audio(audio_path):
    """
    Loads audio from a file using librosa while preserving the original sample rate.
    """
    audio, sr = librosa.load(audio_path, sr=None)
    return audio, sr

# Step 4: Resample the amplitude curve (extracted from the image)
# so that its length matches the audio's number of samples.
def resample_curve_to_audio_length(amplitude_curve, audio_length):
    """
    Linearly interpolates the amplitude curve to have the same number of samples as the audio.
    """
    img_length = len(amplitude_curve)
    if img_length == audio_length:
        return amplitude_curve  # Already matching
    
    x_original = np.linspace(0, 1, img_length)
    x_target = np.linspace(0, 1, audio_length)
    resampled_curve = np.interp(x_target, x_original, amplitude_curve)
    return resampled_curve

# Step 5: Apply the amplitude envelope to the audio sample-by-sample.
def apply_amplitude_envelope(audio, envelope):
    """
    Multiplies the audio signal sample-by-sample by the amplitude envelope.
    Both must have the same length.
    """
    if len(audio) != len(envelope):
        raise ValueError("Audio and envelope lengths must match.")
    return audio * envelope

# Step 6: Save the adjusted audio.
def save_audio(audio, sr, output_path):
    sf.write(output_path, audio, sr)

# (Optional) Step 7: Visualize the process.
def plot_results(image_data, raw_amplitude_curve, envelope, original_audio, adjusted_audio):
    plt.figure(figsize=(12, 10))

    # Plot the original waveform image.
    plt.subplot(4, 1, 1)
    plt.imshow(image_data, cmap='gray', aspect='auto')
    plt.title("Original Waveform Image")

    # Plot the raw extracted amplitude curve (per image column).
    plt.subplot(4, 1, 2)
    plt.plot(raw_amplitude_curve, color='blue')
    plt.title("Extracted 1D Amplitude Curve (Before Resampling)")

    # Plot the resampled envelope (matching audio length).
    plt.subplot(4, 1, 3)
    plt.plot(envelope, color='green')
    plt.title("Resampled Envelope (Matches Audio Length)")

    # Compare original and adjusted audio waveforms.
    plt.subplot(4, 1, 4)
    plt.plot(original_audio, label='Original Audio', alpha=0.5)
    plt.plot(adjusted_audio, label='Adjusted Audio', alpha=0.8)
    plt.title("Comparison of Original vs. Adjusted Audio Waveforms")
    plt.legend()

    plt.tight_layout()
    plt.show()

#############################################
# Example usage
#############################################
if __name__ == "__main__":
    image_path = "/Users/aaravkumar/Desktop/3.295-95.982-9.1-crop.png"
    audio_path = "/Users/aaravkumar/Desktop/eq-shortened.wav"
    output_audio_path = "adjusted_audio.wav"

    # 1) Load the waveform image.
    image_data = load_waveform_image(image_path)

    # 2) Extract the amplitude curve using center-based measurement.
    raw_amplitude_curve = extract_waveform_from_image(image_data)

    # 3) Load the audio.
    audio, sr = load_audio(audio_path)

    # 4) Resample the amplitude curve to match the number of audio samples.
    envelope = resample_curve_to_audio_length(raw_amplitude_curve, len(audio))

    # 5) Apply the envelope to the audio.
    adjusted_audio = apply_amplitude_envelope(audio, envelope)

    # 6) Save the adjusted audio.
    save_audio(adjusted_audio, sr, output_audio_path)

    # 7) (Optional) Plot for visualization.
    plot_results(image_data, raw_amplitude_curve, envelope, audio, adjusted_audio)

    print(f"Adjusted audio saved to: {output_audio_path}")
