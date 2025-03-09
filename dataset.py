from datasets import load_dataset
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display, Markdown, Latex
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)

math500 = load_dataset("HuggingFaceH4/MATH-500")

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(device)
model.eval()

def extract_reasoning_steps(response):
    """
    Extract the chain of thought steps and final answer from the model response.
    
    Args:
        response (str): The model's response containing reasoning steps and answer
        
    Returns:
        tuple: (list of reasoning steps, final answer)
    """
    # Check if the response contains the "</think>" tag
    if "</think>" in response:
        # Split the response into reasoning part and answer part
        reasoning_part, final_answer = response.split("</think>", 1)
        final_answer = final_answer.lstrip()
    else:
        reasoning_part = response
        final_answer = ""
        
    # Split the reasoning part into individual steps
    # Steps are separated by double newlines
    reasoning_steps = reasoning_part.split("\n\n")
    if len(reasoning_steps) > 0:
        reasoning_steps[-1] = reasoning_steps[-1].rstrip()

    return reasoning_steps, final_answer

# Doing this one at a time is slow, but memory is limited and I can just leave it runnig
def process_example(example):
    problem = example["problem"]
    messages = [
        {"role": "user", "content": problem}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    steps, model_answer = extract_reasoning_steps(response)

    return {
        "prompt": problem,
        "steps": steps,
        "answer": example["answer"],
        "model_answer": model_answer,
        "raw_response": response
    }

dataset = math500["test"].map(process_example)
dataset.save_to_disk("math500-steps.hf")