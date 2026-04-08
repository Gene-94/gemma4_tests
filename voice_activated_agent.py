import sounddevice, soundfile, numpy, mlx_whisper, torch
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from silero_vad import load_silero_vad, VADIterator

def listen():
    model = load_silero_vad()

    vad_iterator = VADIterator(model, sampling_rate=16000, threshold=0.65, min_silence_duration_ms=1500)

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

    return "./voice_prompt.wav"
            

def transcribe_audio(filepath):
    result = mlx_whisper.transcribe(
        filepath,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo"
    )
    return result["text"].strip()

def think(transcribed_input):
   
    model, processor = load("mlx-community/gemma-4-26B-A4B-it-heretic-4bit",
        chat_template_kwargs={"enable_thinking": True},)


    prompt = apply_chat_template(
        processor, model.config, transcribed_input
    )

    result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        max_tokens=1000,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    print("\n"+result.text)

if __name__ == "__main__":
    while True:
        think(transcribe_audio(listen()))