This repo contains code for some basic interpretability experiments with the Qwen-1.5B distill of DeepSeek R1. I was trying to investigate how the model decides when to go back and check its work, or conclude the reasoning trace.

Project overview:
- Generate a dataset of reasoning traces (`dataset.py`)
- Label them with Claude (`label.ipynb`)
- Blackbox analysis of how often the model makes the same decision when we regenerate the rest of the trace (`blackbox.ipynb`)
- Find a steering vector to control this behavior (`tl.ipynb`)
