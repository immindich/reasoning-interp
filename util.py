def reconstruct_prompt(tokenizer, problem, n_steps=None):
    messages = [
        {"role": "user", "content": problem["problem"]}
    ]
    initial_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if n_steps is None:
        n_steps = len(problem["steps"])

    reasoning_steps = "\n\n".join(problem["steps"][:n_steps])

    return initial_prompt + reasoning_steps + "\n\n"
