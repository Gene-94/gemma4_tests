from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/gemma-4-26B-A4B-it-heretic-4bit",
    chat_template_kwargs={"enable_thinking": True},)

user_input = input("What's up doc? \n\n")

prompt = apply_chat_template(
    processor, model.config, user_input
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print("\n"+result.text)