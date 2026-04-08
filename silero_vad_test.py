import sounddevice, numpy, torch, soundfile
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator


model = load_silero_vad()

vad_iterator = VADIterator(model, sampling_rate=1600, threshold=0.6, min_silence_duration_ms=1500)

stream = sounddevice.InputStream(samplerate=16000, channels=1, dtype='float32')

audio_buffer = []

recording = False

print("Mic open")

with stream:
    while True:
        chunk, _ = stream.read(512)

        chunk_tensor = torch.from_numpy(chunk.flatten())

        speech = vad_iterator(chunk_tensor, return_seconds=False)

        if speech:
            if "start" in speech:
                recording = True
                audio_buffer = []
                audio_buffer.append(chunk)
                print("detected voice. Listening...")

            elif "end" in speech:
                audio_buffer.append(chunk)
                recording = False
                print("I guess you're done")
                break
        
        elif recording:
            audio_buffer.append(chunk)

full_speech = numpy.concatenate(audio_buffer, axis=0)

soundfile.write("voice_prompt.wav", full_speech, 16000)

print("Saved voice segment")