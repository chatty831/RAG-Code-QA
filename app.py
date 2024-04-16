from huggingface_hub import InferenceClient
import gradio as gr
from hf import *

client = InferenceClient(
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    token='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'
    # "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
)

def fetch_docs(prompt,documentation):
    docs = None
    if documentation=='Huggingface':
        docs = hf_docs(prompt)
    return docs

def format_prompt(message, history):
  prompt = "<|im_start|>user\n"
  for user_prompt, bot_response in history:
    prompt += f"{user_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n{bot_response}<|im_end|>"
  prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
  return prompt

def generate(
    prompt, history, documentation, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        # seed=42,
    )
    
    docs = ''
    prompt = docs + "\n\nThis is the question. Make sure the answer is correct.\nQuestion\n" + prompt 
    
    if documentation:
        docs = fetch_docs(prompt,documentation)
        docs = f'''This the relevant documentation of {documentation} for the question asked below. Refer to the below documentation and answer the question in the end.\n''' + docs
        prompt = docs + "\n\nThis is the question, Please answer it after referring to the above docs. Make sure the answer is correct.\nQuestion\n" + prompt 
    
    formatted_prompt = format_prompt(f"{prompt}", history)
    # print(formatted_prompt)
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


additional_inputs=[
    gr.Dropdown(
        # label="System Prompt",
        choices=['Huggingface', 'PyTorch', 'langchain', 'gradio', 'nextjs'],
        label="Documentations",
        value=None,
        # max_lines=1,
        # interactive=True,
    ),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

examples=[["I'm planning a vacation to Japan. Can you suggest a one-week itinerary including must-visit places and local cuisines to try?", None, None, None, None, None, ],
          ["Can you write a short story about a time-traveling detective who solves historical mysteries?", None, None, None, None, None,],
          ["I'm trying to learn French. Can you provide some common phrases that would be useful for a beginner, along with their pronunciations?", None, None, None, None, None,],
          ["I have chicken, rice, and bell peppers in my kitchen. Can you suggest an easy recipe I can make with these ingredients?", None, None, None, None, None,],
          ["Can you explain how the QuickSort algorithm works and provide a Python implementation?", None, None, None, None, None,],
          ["What are some unique features of Rust that make it stand out compared to other systems programming languages like C++?", None, None, None, None, None,],
         ]

gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel"),
    additional_inputs=additional_inputs,
    title="Mixtral 46.7B",
    examples=examples,
    concurrency_limit=20,
).launch(show_api=False)