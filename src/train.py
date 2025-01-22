import torch
from dataclasses import dataclass
from data_handler import DataHandler
from bpe_tokenizer import BPETokenizer
from model import GPT

@dataclass
class GPT_ModelConfig:
    vocab_size = 1000
    batch_size: int = 128
    block_size: int = 256 # context length
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1
    bias: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class GPT_TrainConfig:
    learning_rate: float = 3e-4
    max_iters: int = 8000
    eval_interval: int = 500
    eval_iters: int = 200
    checkpoint_interval = 100

    # for optimizer
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

@torch.no_grad()
def estimate_loss(model, data_handler, cfg, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_handler.get_batch(split, cfg)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()

    return out

def train_gpt_model(model, data_handler, cfg, train_cfg, checkpoint_path, weights_save_path, from_checkpoint):

    optimizer = model.configure_optimizers(train_cfg.learning_rate, train_cfg.weight_decay, betas=(train_cfg.beta1, train_cfg.beta2), device_type=cfg.device)

    start_iter = -1
    if from_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint["iter"]
        print("training from the checkpoint")

    for iter in range(train_cfg.max_iters):
        if iter > start_iter:
            if iter % train_cfg.eval_interval == 0 or iter == train_cfg.max_iters-1:
                losses = estimate_loss(model, data_handler, cfg, train_cfg.eval_iters)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = data_handler.get_batch('train', cfg)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter % train_cfg.checkpoint_interval == 0 or iter == train_cfg.max_iters-1:

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': iter,
                }, checkpoint_path)

                print(f"iter {iter}: the checkpoint was saved successfully")



    torch.save(model.state_dict(), weights_save_path)
    

cfg = GPT_ModelConfig()
train_cfg = GPT_TrainConfig()

data_handler = DataHandler()
data_handler.load_dir('../data')

tokenizer = BPETokenizer()
tokenizer.train(data_handler.text_data, vocab_size=cfg.vocab_size, verbose=True)
tokenizer.save_merges(f"../model-data/merges/merges-{cfg.vocab_size - 256}.json")
tokenizer.save_vocab(f"../model-data/vocabs/vocab-{len(tokenizer.vocab)}.json")
# or
# tokenizer.load_merges("../model-data/merges/merges-744.json")
# tokenizer.load_vocab("../model-data/vocabs/vocab-1000.json")

data_handler.data_to_tokens(tokenizer.encode)
data_handler.train_val_split(train_size=0.9)

model = GPT(cfg)
model = model.to(cfg.device)

train_gpt_model(model, data_handler, cfg, train_cfg, '../model-data/checkpoints/checkpoint.pth', '../model-data/weights/weights.pth', False)