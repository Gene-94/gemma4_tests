import sounddevice, soundfile, numpy, mlx_whisper

def get_audio():
    sample_rate = 16000
    duration = 5
    channels = 1

    print(f"Recording for {duration} seconds...")

    audio_data = sounddevice.rec(
        int(duration * sample_rate),
        channels=channels,
        dtype='float32'
    )

    sounddevice.wait()
    print("done recording ")

    filename = "test_audio.wav"
    soundfile.write(filename, audio_data, sample_rate)

    return filename

def transcribe_audio(filename):
    result = mlx_whisper.transcribe(
        "./"+filename,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
    )
    return result["text"].strip()

if __name__ == "__main__":
    scrib = transcribe_audio(get_audio())
    print(f"So you '{scrib}', is that it?")