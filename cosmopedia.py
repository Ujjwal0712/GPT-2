"""
Cosmopedia dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "cosmopedia".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "cosmopedia"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
def process_dataset():
    """Downloads the dataset and tokenizes it in parallel."""
    fw = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

    nprocs = max(1, os.cpu_count() // 2)  # Use half of the available CPUs
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # Preallocate buffer
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                # Append tokens to the current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)

                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"cosmopedia_{split}_{shard_index:06d}.npy")

                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)

                shard_index += 1
                progress_bar = None

                # Populate the next shard with leftover tokens
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"cosmopedia_{split}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows
    process_dataset()