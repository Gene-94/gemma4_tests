import sounddevice, soundfile, numpy

def test_mic():
    sample_rate = 16000
    duration = 10
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

if __name__ == "__main__":
    test_mic()