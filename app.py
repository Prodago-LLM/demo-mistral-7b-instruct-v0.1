from potassium import Potassium, Request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import pandas as pd

import evaluate
from evaluate import logging

app = Potassium("mistral-7b-instruct-v0.1")

@app.init
def init() -> dict:
    """Initialize the application with the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    context = {
        "model": model,
        "tokenizer": tokenizer
    }
    return context

@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate text from a prompt."""
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    max_new_tokens = request.json.get("max_new_tokens", 512)
    prompt = request.json.get("prompt")
    device = "cuda"
    text = prompt
    encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    result = decoded[0]
    return Response(json={"outputs": result}, status=200)


@app.handler("/evaluate")
def perplexityScore(context: dict, request: Request) -> Response:
    """
    Calculate the perplexity score for a given model.
    """
    batch_size = 16
    add_start_token = True
    device = None
    max_length = None

    df = pd.read_csv('OP_mapping.csv')
    input_texts = df['OP'].tolist()[:50]
    input_texts = [s for s in input_texts if s != '']
    predictions = input_texts

    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = context.get("model")
    model = model.to(device)
    tokenizer = context.get("tokenizer")

    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0{"outputs": result}
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return Response(json={"perplexities": ppls, "mean_perplexity": np.mean(ppls)}, status=200)


if __name__ == "__main__":
    app.serve()