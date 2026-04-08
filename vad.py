import sounddevice, soundfile, numpy, mlx_whisper
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

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
    scrib = transcribe_audio(get_audio())
    think(scrib)