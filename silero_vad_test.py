import sounddevice, numpy, torch, soundfile
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


model = load_silero_vad()

wav = read_audio("./test_audio.wav")

timestamps = get_speech_timestamps(wav, model, return_seconds=True)

print(timestamps)