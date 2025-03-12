import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import time
from transformers import GPT2LMHeadModel
import math
import inspect
import os
from hellaswag import render_example, iterate_examples

def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    
    if not checkpoints:
        print("No checkpoint found, starting training from scratch.")
        return 0  # Start from step 0

    latest_checkpoint = sorted(checkpoints)[-1]  # Get latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng_state'])
    
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    start_step = checkpoint['step']
    print(f"Resumed training from step {start_step}")
    
    return start_step

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #Query, Key and Value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        #Output Projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #Masking
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)

        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1,2)
      
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.bias[:, :, T-1, T-1] == 0, float('inf'))
        # attn = F.softmax(attn, dim = 1)
        # y = attn @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        #Output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd), 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) 

        #Weight Sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std  = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        #Start with all the parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        #Create optim groups, weight decay the parameter that have dim > 1, otherwise no weight decay
        #All weigh tensors in matmul + embeddings decay, all biases and layernorms don't decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 ]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nondecay_params, 'weight_decay' : 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        print(f"number of parameters: {num_decay_params} weight decay, {num_nondecay_params} no weight decay")  
        print(f"number of parameters: {num_decay_params + num_nondecay_params} total")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused_adam = fused_available and 'cuda' in device
        print(f"using fused adam: {use_fused_adam}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8)
        return optimizer


    def forward(self, idx, targets = None):
        #idx is of the shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size
        
        #Forward the positional and token embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        #Forward the decoder blocks
        for block in self.transformer.h:
            x = block(x)

        #Final layer normalization and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss
    

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

#-------------------------------------------

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T

        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "cosmopedia"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
       
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T 

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T  + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T 
        return x, y
    
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
    
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print(torch.cuda.get_device_name(device))
    print("Compute Capability:", torch.cuda.get_device_capability(device))
    print("Total SMs:", torch.cuda.get_device_properties(device).multi_processor_count)
print(f"Using device: {device}")
torch.manual_seed(1337)

#Gradient accumulation
total_batch_size = 8192
B = 2
T = 1024
assert total_batch_size % (B*T) == 0
grad_accum_steps = total_batch_size // (B*T)
print(f"gradient accumulation: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 3814
max_steps = 76292
def get_lr(it):
    # 1) Linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it+1)/ warmup_steps
    
    # 2) If it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * ( max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps = 1e-8)

optimizer = model.configure_optimizer(weight_decay = 0.1, learning_rate = 3e-4, device = device)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):   
    t0 = time.time()
    last_step = (step == max_steps - 1)
    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),  # Save model parameters
                    'optimizer': optimizer.state_dict(),  # Save optimizer state
                    'config': model.config,  # Save model configuration
                    'step': step,  # Save current training step
                    'val_loss': val_loss_accum.item(),  # Save validation loss
                    'rng_state': torch.get_rng_state(),  # Save RNG state for reproducibility
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at step {step} -> {checkpoint_path}")

    if (step % 250 == 1 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm / num_total
       
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  

    #Determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_per_sec= (train_loader.B * train_loader.T) / (t1 -t0) * grad_accum_steps
    print(f"iteration {step:4d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt:.2f}ms, tokens/sec: {tokens_per_sec:.2f}t/s")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")





