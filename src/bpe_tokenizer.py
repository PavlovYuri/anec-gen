import base64
import json

class BPETokenizer():
    def __init__(self):
        pass

    def get_stats(self, ids, counts=None):
      counts = {} if counts is None else counts
      for pair in zip(ids, ids[1:]):
          counts[pair] = counts.get(pair, 0) + 1

      return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids)-1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1

        return newids

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def save_merges(self, filepath):
        merges = {str(k): str(v) for k, v in self.merges.items()}
        with open(filepath, 'w') as f:
            json.dump(merges, f)

    def load_merges(self, filepath):
        with open(filepath) as f:
            merges = json.load(f)

        self.merges = {tuple(map(int, k[1:-1].split(', '))): int(v) for k, v in merges.items()}

    def save_vocab(self, filepath):
        vocab = {k: base64.b64encode(v).decode('utf-8') for k, v in self.vocab.items()}

        with open(filepath, "w") as f:
            json.dump(vocab, f)

    def load_vocab(self, filepath):
        with open(filepath, "r") as f:
            vocab = json.load(f)

        self.vocab = {int(k): base64.b64decode(v) for k, v in vocab.items()}

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)

        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")

        return text