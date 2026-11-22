"""
Stateless, batchable streaming for Nemotron Speech Streaming (RNN-T).

The idea
--------
`model.generate()` keeps each user's state inside one call (and flags on the
model object), so one model copy can serve only one user. Here every piece of
per-user state lives in a `StreamState` object OUTSIDE the model:

    encoder attention cache   keys/values [B, heads, <=70, dim] per layer + a position counter
    encoder conv cache        [B, channels, kernel-1] per conv layer
    decoder LSTM state        hidden/cell [layers, B, dim], last output [B, 1, dim]

The model becomes a pure function:   (chunk, state) -> (new tokens, new state)

and `BatchedStreamer.step()` can run many users through ONE model call by
stacking their states on the batch dimension, then splitting them back.

Phases covered by this file
---------------------------
Phase 1: `step([one_state], ...)`  reproduces model.generate() exactly.
Phase 2: `step([s1, s2, ...], ...)` batches users whose caches are the same size
         (e.g. they started together, or both have a full 5.6 s history).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Per-user state (lives outside the model)
# ---------------------------------------------------------------------------
@dataclass
class StreamState:
    encoder_cache: object | None = None     # transformers DynamicCache (sliding window)
    padding_cache: object | None = None     # conv history cache
    dec_out: torch.Tensor | None = None     # [1, 1, D] last decoder output
    dec_h: torch.Tensor | None = None       # [layers, 1, D]
    dec_c: torch.Tensor | None = None       # [layers, 1, D]
    tokens: list = field(default_factory=list)   # emitted (non-blank) token ids
    chunks_done: int = 0

    @property
    def started(self) -> bool:
        return self.encoder_cache is not None

    def cache_frames(self) -> int:
        """Frames of attention history held (grows 7/chunk, caps at the sliding window)."""
        return 0 if self.encoder_cache is None else self.encoder_cache.layers[0].keys.shape[2]

    def cache_counter(self) -> int:
        """Total frames ever seen; used for positions inside the encoder."""
        return 0 if self.encoder_cache is None else self.encoder_cache.get_seq_length()


# ---------------------------------------------------------------------------
# Stack / split cache objects along the batch dimension
# ---------------------------------------------------------------------------
def _merge_obj(objs: list):
    """Copy objs[0]'s structure; concatenate every per-user tensor on dim 0.

    Non-tensor attributes (ints, flags) must be identical across users, otherwise
    the users can't share one batch and we raise.
    """
    merged = copy.copy(objs[0])
    for name, val in vars(objs[0]).items():
        if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == 1:
            merged.__dict__[name] = torch.cat([o.__dict__[name] for o in objs], dim=0)
        elif isinstance(val, torch.Tensor):
            for o in objs[1:]:
                if not torch.equal(o.__dict__[name], val):
                    raise ValueError(f"Cannot batch: '{name}' differs between users")
        elif isinstance(val, (int, float, bool, str)) or val is None:
            for o in objs[1:]:
                if o.__dict__[name] != val:
                    raise ValueError(f"Cannot batch: '{name}' differs ({o.__dict__[name]} vs {val})")
    return merged


def _split_obj(merged, n: int) -> list:
    outs = []
    for i in range(n):
        o = copy.copy(merged)
        for name, val in vars(merged).items():
            if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == n:
                o.__dict__[name] = val[i:i + 1].clone()   # each user owns its memory
        outs.append(o)
    return outs


def merge_encoder_caches(caches: list):
    merged = copy.copy(caches[0])
    merged.layers = [_merge_obj([c.layers[i] for c in caches]) for i in range(len(caches[0].layers))]
    return merged


def split_encoder_cache(merged, n: int) -> list:
    per_layer = [_split_obj(layer, n) for layer in merged.layers]
    outs = []
    for i in range(n):
        c = copy.copy(merged)
        c.layers = [per_layer[l][i] for l in range(len(per_layer))]
        outs.append(c)
    return outs


def merge_padding_caches(caches: list):
    merged = copy.copy(caches[0])
    merged.layers = {k: _merge_obj([c.layers[k] for c in caches]) for k in caches[0].layers}
    return merged


def split_padding_cache(merged, n: int) -> list:
    per_key = {k: _split_obj(v, n) for k, v in merged.layers.items()}
    outs = []
    for i in range(n):
        c = copy.copy(merged)
        c.layers = {k: per_key[k][i] for k in per_key}
        outs.append(c)
    return outs


# ---------------------------------------------------------------------------
# The batched streaming step
# ---------------------------------------------------------------------------
class BatchedStreamer:
    """Runs one chunk for a group of users in one model call.

    align_full_caches (Phase 3 preview, off by default): once users have a FULL
    5.6 s history, their caches are the same size but their position counters
    differ. The mask works in whole chunks and positions are relative, so the
    counter should not change the math; with this flag on, such users are
    batched together by temporarily giving them a common counter.
    """

    def __init__(self, model, num_lookahead_tokens: int = 6, align_full_caches: bool = False):
        self.model = model
        self.align_full_caches = align_full_caches
        self.L = num_lookahead_tokens
        self.blank = model.config.blank_token_id
        self.max_symbols = model.max_symbols_per_step
        sub = model.config.encoder_config.subsampling_factor
        self.first_chunk_frames = 1 + sub * self.L          # mel frames, first chunk
        self.chunk_frames = sub * (self.L + 1)               # mel frames, later chunks

    # ---- which users can share a batch ------------------------------------
    def window(self) -> int:
        return self.model.config.encoder_config.sliding_window - 1        # 70 frames = 5.6 s

    def group_key(self, state: StreamState):
        """Users with the same key can be stacked into one model call."""
        if not state.started:
            return ("first",)
        frames = state.cache_frames()
        if self.align_full_caches and frames >= self.window():
            return ("full",)
        return ("cont", frames, state.cache_counter())

    def group(self, states: list[StreamState]) -> list[list[int]]:
        """Split users (by index) into batch-compatible groups."""
        groups: dict = {}
        for i, s in enumerate(states):
            groups.setdefault(self.group_key(s), []).append(i)
        return list(groups.values())

    # ---- decoder helpers -------------------------------------------------
    def _decoder_start(self, batch: int, device, dtype):
        """Decoder state before any token: LSTM run once on <blank> from zeros (as generate() does)."""
        dec = self.model.decoder
        ids = torch.full((batch, 1), self.blank, dtype=torch.long, device=device)
        layers, hidden = dec.lstm.num_layers, dec.lstm.hidden_size
        h0 = torch.zeros(layers, batch, hidden, device=device, dtype=dtype)
        c0 = torch.zeros_like(h0)
        out, (h, c) = dec.lstm(dec.embedding(ids), (h0, c0))
        return dec.decoder_projector(out), h, c

    def _decoder_advance(self, tokens, dec_out, h, c, mask):
        """Feed emitted tokens to the LSTM; only rows in `mask` change (matches generate())."""
        dec = self.model.decoder
        out, (h2, c2) = dec.lstm(dec.embedding(tokens[:, None]), (h, c))
        out = dec.decoder_projector(out)
        md, mh = mask.view(-1, 1, 1), mask.view(1, -1, 1)
        return torch.where(md, out, dec_out), torch.where(mh, h2, h), torch.where(mh, c2, c)

    # ---- one streaming step for a group of users -------------------------
    @torch.inference_mode()
    def step(self, states: list[StreamState], chunks: list[torch.Tensor]) -> list[list[int]]:
        """Run one chunk for each user in ONE model call. Returns new tokens per user.

        All users in a call must be in the same phase: all on their first chunk,
        or all continuing with caches of the same size.
        """
        n = len(states)
        first = not states[0].started
        if any(s.started == first for s in states):
            raise ValueError("Cannot mix users on their first chunk with continuing users")
        need = self.first_chunk_frames if first else self.chunk_frames
        for ch in chunks:
            if ch.shape[1] != need:
                raise ValueError(f"Chunk has {ch.shape[1]} mel frames, expected {need}")

        m = self.model
        x = torch.cat(chunks, dim=0).to(m.device, m.dtype)
        sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
        t0 = time.perf_counter()

        # ---------------- encoder: one call for all users ----------------
        counters = None
        if first:
            enc_cache = pad_cache = None
        else:
            if self.align_full_caches and len({s.cache_counter() for s in states}) > 1:
                if any(s.cache_frames() < self.window() for s in states):
                    raise ValueError("Counters differ and not all caches are full; use group() first")
                # Give everyone the first user's counter for this call; restored after the split.
                counters = [s.cache_counter() for s in states]
                for st in states:
                    for layer in st.encoder_cache.layers:
                        layer.cumulative_length = counters[0]
            enc_cache = merge_encoder_caches([s.encoder_cache for s in states])
            pad_cache = merge_padding_caches([s.padding_cache for s in states])
        out = m.get_audio_features(input_features=x, cache=enc_cache, padding_cache=pad_cache,
                                   num_lookahead_tokens=self.L, use_cache=True,
                                   output_attention_mask=False)
        frames = out.pooler_output                       # [B, T, D]
        sync()
        t1 = time.perf_counter()
        enc_caches = split_encoder_cache(out.cache, n)
        pad_caches = split_padding_cache(out.padding_cache, n)
        if counters is not None:
            grown = enc_caches[0].get_seq_length() - counters[0]
            for c, own in zip(enc_caches, counters):
                for layer in c.layers:
                    layer.cumulative_length = own + grown

        # ---------------- decoder: batched greedy RNN-T ----------------
        if first:
            dec_out, h, c = self._decoder_start(n, frames.device, frames.dtype)
        else:
            dec_out = torch.cat([s.dec_out for s in states], dim=0)
            h = torch.cat([s.dec_h for s in states], dim=1)
            c = torch.cat([s.dec_c for s in states], dim=1)

        B, T, _ = frames.shape
        idx = torch.zeros(B, dtype=torch.long, device=frames.device)   # frame pointer per user
        symbols = torch.zeros_like(idx)
        rows = torch.arange(B, device=frames.device)
        new_tokens: list[list[int]] = [[] for _ in range(B)]

        while bool((idx < T).any()):
            active = idx < T
            enc_t = frames[rows, idx.clamp(max=T - 1)]                  # [B, D]
            logits = m.joint.head(m.joint.activation(enc_t + dec_out[:, 0]))
            tok = logits.argmax(dim=-1)
            blank = tok == self.blank

            emit = active & ~blank
            for b in torch.nonzero(emit).flatten().tolist():
                new_tokens[b].append(int(tok[b]))

            # RNN-T rule: advance on blank; stay on a token; force advance after max_symbols.
            symbols = torch.where(blank, torch.zeros_like(symbols), symbols + 1)
            force = symbols >= self.max_symbols
            symbols = torch.where(blank | force, torch.zeros_like(symbols), symbols)
            idx = idx + ((blank | force) & active).long()

            if bool(emit.any()):
                dec_out, h, c = self._decoder_advance(tok, dec_out, h, c, emit)

        sync()
        t2 = time.perf_counter()
        self.last_timing = {"encoder_ms": (t1 - t0) * 1000, "decoder_ms": (t2 - t1) * 1000, "users": n}

        # ---------------- split state back to each user ----------------
        for i, s in enumerate(states):
            s.encoder_cache, s.padding_cache = enc_caches[i], pad_caches[i]
            s.dec_out = dec_out[i:i + 1].clone()
            s.dec_h, s.dec_c = h[:, i:i + 1].clone(), c[:, i:i + 1].clone()
            s.tokens.extend(new_tokens[i])
            s.chunks_done += 1
        return new_tokens


# ---------------------------------------------------------------------------
# Audio -> exact-size mel chunks (same slicing as the model card / engines.py)
# ---------------------------------------------------------------------------
def mel_chunks(feature_extractor, audio, num_lookahead_tokens: int, subsampling: int = 8,
               pad_final: bool = True):
    """Yield [1, frames, mels] chunks of exactly the sizes the model requires."""
    import numpy as np
    hop = feature_extractor.hop_length
    n_fft = feature_extractor.n_fft
    win = feature_extractor.win_length
    first_frames = 1 + subsampling * num_lookahead_tokens
    per_frames = subsampling * (num_lookahead_tokens + 1)
    first_samples = (first_frames - 1) * hop + win // 2
    per_samples = per_frames * hop + win
    step = per_frames * hop

    def feats(x, first):
        f = feature_extractor(x, sampling_rate=16000, center=first, return_tensors="pt")["input_features"]
        return f[:, :first_frames] if first else f

    audio = np.asarray(audio, dtype=np.float32)
    total = len(audio)
    a = np.concatenate([audio, np.zeros(first_samples + per_samples, dtype=np.float32)])
    yield feats(a[:first_samples], True)
    mel = first_frames
    while True:
        start = mel * hop - n_fft // 2
        if start >= total + (step if pad_final else 0):
            return
        end = start + per_samples
        if end > len(a):
            a = np.concatenate([a, np.zeros(end - len(a), dtype=np.float32)])
        lead = max(0, -start)
        piece = a[max(start, 0):end]
        if lead:
            piece = np.concatenate([np.zeros(lead, dtype=np.float32), piece])
        yield feats(piece, False)
        mel += per_frames


class MelChunker:
    """Live version of mel_chunks(): audio arrives in pieces, chunks come out when ready.

    Produces exactly the same chunks as mel_chunks() on the complete audio, however
    the audio is split. Call add() as audio arrives, pop() to get ready chunks,
    finish() when the user stops (pads with silence and flushes the last chunk).
    """

    def __init__(self, feature_extractor, num_lookahead_tokens: int, subsampling: int = 8):
        import numpy as np
        self.np = np
        self.fe = feature_extractor
        self.hop = feature_extractor.hop_length
        self.n_fft = feature_extractor.n_fft
        win = feature_extractor.win_length
        self.first_frames = 1 + subsampling * num_lookahead_tokens
        self.per_frames = subsampling * (num_lookahead_tokens + 1)
        self.first_samples = (self.first_frames - 1) * self.hop + win // 2
        self.per_samples = self.per_frames * self.hop + win
        self.step = self.per_frames * self.hop
        self.buf = np.zeros(0, dtype=np.float32)
        self.base = 0            # absolute sample index of buf[0]
        self.total = 0           # samples received
        self.mel = None          # next mel frame index; None = first chunk not produced yet
        self.final = False

    def add(self, audio) -> None:
        a = self.np.asarray(audio, dtype=self.np.float32)
        self.buf = self.np.concatenate([self.buf, a])
        self.total += len(a)

    def finish(self) -> None:
        self.final = True

    @property
    def done(self) -> bool:
        """True when finished and every chunk (including the flush chunk) has been produced."""
        if not self.final:
            return False
        if self.mel is None:
            return self.total == 0
        return self.mel * self.hop - self.n_fft // 2 >= self.total + self.step

    def _take(self, start: int, end: int):
        np = self.np
        lead = max(0, -start)
        lo, hi = max(start, 0) - self.base, end - self.base
        piece = self.buf[lo:min(hi, len(self.buf))]
        tail = max(0, (end - max(start, 0)) - len(piece))
        parts = [np.zeros(lead, np.float32), piece, np.zeros(tail, np.float32)]
        return np.concatenate(parts)

    def _feats(self, x, first: bool):
        f = self.fe(x, sampling_rate=16000, center=first, return_tensors="pt")["input_features"]
        return f[:, :self.first_frames] if first else f

    def pop(self) -> list:
        out = []
        while True:
            if self.mel is None:
                if self.total == 0 or (self.total < self.first_samples and not self.final):
                    return out
                out.append(self._feats(self._take(0, self.first_samples), True))
                self.mel = self.first_frames
                continue
            start = self.mel * self.hop - self.n_fft // 2
            end = start + self.per_samples
            if self.final:
                if start >= self.total + self.step:
                    return out
            elif end > self.total:
                return out
            out.append(self._feats(self._take(start, end), False))
            self.mel += self.per_frames
            cut = (self.mel * self.hop - self.n_fft // 2) - self.base               # next chunk's start
            if cut > 16000:                                                          # trim in 1 s steps
                self.buf = self.buf[cut:]
                self.base += cut
