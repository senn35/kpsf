import os
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import tensorflow as tf
import sounddevice as sd
import wave
import json
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
DURATION = 5
TF_MODEL_PATH = "audio_classifier.h5"
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
WAV_FILENAME = "recording.wav"
GAIN_FACTOR = 10
def compute_mfcc(signal, sample_rate=48000, num_mfcc=13):
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_size = 0.025
    frame_stride = 0.01
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    padded_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
    ).T
    frames = padded_signal[indices.astype(np.int32)]
    frames *= np.hamming(frame_length)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin_points[m - 1])
        f_m = int(bin_points[m])
        f_m_plus = int(bin_points[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_mfcc]
    return np.mean(mfcc, axis=0)
def add_white_noise(signal, noise_level=0.005):
    """Adds white noise to the signal."""
    noise = np.random.normal(0, 1, len(signal)) * noise_level * np.max(np.abs(signal))
    return signal + noise

def is_silence(signal, threshold=100):
    """Check if the audio signal is silence based on an amplitude threshold."""
    return np.max(np.abs(signal)) < threshold

def record_audio(filename: str, duration: int, sample_rate: int):
    print("Recording audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    
    # Convert to floating-point for safer multiplication
    audio_data = audio_data.astype(np.float32)
    # Apply gain
    boosted_audio = audio_data * GAIN_FACTOR
    boosted_audio = np.clip(boosted_audio, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
    audio_data = boosted_audio
    

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    print(f"Recording saved to {filename}")
    return audio_data

def classify_audio(model_path: str, audio_path: str, signal):
    if is_silence(signal):
        print("Silence detected. Skipping classification.")
        signal = add_white_noise(signal)
        return -1  # Indicator for silence

    model = tf.keras.models.load_model(model_path)
    sample_rate, signal = wavfile.read(audio_path)
    if signal.ndim > 1:
        signal = signal[:, 0]
    mfcc_features = compute_mfcc(signal, sample_rate).reshape(1, -1)
    prediction = model.predict(mfcc_features)
    probability = prediction[0][0]
    predicted_class = int(probability > 0.8)
    print(f"Predicted probability: {probability:.2f}")
    print(f"Predicted class: {predicted_class} ({'scream' if predicted_class == 1 else 'non_scream'})")
    return predicted_class

def transcribe_audio(filename: str):
    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError("Vosk model not found. Please set the correct path.")
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    with wave.open(filename, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                break
    result = json.loads(rec.Result())
    transcription = result.get('text', '')
    print(f"Transcription: {transcription}")
    return transcription

def main():
    while True:
        audio_data = record_audio(WAV_FILENAME, DURATION, SAMPLE_RATE)
        
        # Classify with TensorFlow
        predicted_class = classify_audio(TF_MODEL_PATH, WAV_FILENAME, audio_data)
        
        # Skip processing if it's silence
        if predicted_class == -1:
            continue
        
        # Transcribe with Vosk
        transcription = transcribe_audio(WAV_FILENAME)

        if predicted_class == 1:
            print("Scream detected! Exiting...")
            break

if __name__ == "__main__":
    main()
