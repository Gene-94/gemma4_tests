from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e4b-it")

prompt = apply_chat_template(
    processor, model.config, "Transcribe this audio",
    num_audios=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    audio=["./audio_message.wav"],
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)