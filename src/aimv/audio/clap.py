"""CLAP audio/text embeddings.

Default model: a LAION CLAP checkpoint trained with additional music data.
"""
from __future__ import annotations

import numpy as np

# laion/larger_clap_music on the HF Hub loads with untrained weights (all norm
# layers at init, every input gets ~the same embedding), so use the general one
DEFAULT_MODEL = "laion/larger_clap_general"


def _to_numpy(out):
    import torch

    if not isinstance(out, torch.Tensor): 
        for attr in ("text_embeds", "audio_embeds", "pooler_output"):
            if getattr(out, attr, None) is not None:
                out = getattr(out, attr)
                break
    return out.detach().cpu().float().numpy()


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


class ClapScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(model_name).eval()
        # nn.Module.to moves in place; PreTrainedModel.to is @wraps-decorated, which breaks Pylance's typing
        torch.nn.Module.to(self.model, self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self.sr = self.processor.feature_extractor.sampling_rate
        try:
            self.logit_scale = float(self.model.logit_scale_a.exp().item())
        except Exception:
            self.logit_scale = 100.0
        self._text_cache: dict[str, np.ndarray] = {}

    def embed_audio(self, windows: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        import torch

        embs = []
        for i in range(0, len(windows), batch_size):
            feats = self.processor.feature_extractor(
                [w for w in windows[i:i + batch_size]], sampling_rate=self.sr, return_tensors="pt")
            feats = {k: v.to(self.device) for k, v in feats.items()}
            with torch.no_grad():
                embs.append(_to_numpy(self.model.get_audio_features(**feats)))
        return l2norm(np.concatenate(embs))

    def embed_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        todo = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            tok = self.processor.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.no_grad():
                emb = l2norm(_to_numpy(self.model.get_text_features(**tok)))
            self._text_cache.update(zip(chunk, emb))
        return np.stack([self._text_cache[t] for t in texts])


def track_embedding(window_embs: np.ndarray) -> np.ndarray:
    return l2norm(window_embs.mean(axis=0))


def zero_shot_probs(audio_emb: np.ndarray, text_embs: np.ndarray, logit_scale: float = 100.0) -> np.ndarray:
    """Softmax over candidate texts (rows of ``text_embs``) for one audio embedding."""
    logits = logit_scale * (text_embs @ audio_emb)
    logits = logits - logits.max()
    p = np.exp(logits)
    return p / p.sum()
