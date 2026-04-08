import mlx_whisper

audio_file = "./audio_message.mp4"

print("Transcribing audio...")

result = mlx_whisper.transcribe(
    audio_file,
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo" 
)

transcribed_text = result["text"].strip()
print(f"You said: {transcribed_text}")