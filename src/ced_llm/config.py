"""Configuration for the minimal dense Causal Encoder-Decoder (CED) LLM."""

from dataclasses import dataclass


@dataclass
class CEDConfig:
    """Hyperparameters for the dense CED language model.

    Attributes:
        vocab_size: Vocabulary size (number of token embeddings / LM head outputs).
        d_model: Model width (must be divisible by ``nhead``).
        n_enc_layers: Number of causal Transformer blocks in the encoder (>= 1).
        n_dec_layers: Number of decoder layers (>= 1).
        nhead: Number of dense multi-head attention heads.
        dim_ff: Feed-forward hidden dimension.
        max_seq_len: Maximum supported sequence length (learned positional table size).
        dropout: Dropout probability in [0, 1).
        pad_token_id: Token id treated as padding (ignored by the LM loss).
        layer_norm_eps: Epsilon for all LayerNorms.
    """

    vocab_size: int = 50257
    d_model: int = 256
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    nhead: int = 8
    dim_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.1
    pad_token_id: int = 50256
    layer_norm_eps: float = 1e-5

    def validate(self) -> None:
        """Validate the configuration, raising ``ValueError``/``TypeError`` on bad values."""
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive int, got %r" % (self.vocab_size,))
        if not isinstance(self.d_model, int) or self.d_model <= 0:
            raise ValueError("d_model must be a positive int, got %r" % (self.d_model,))
        if not isinstance(self.n_enc_layers, int) or self.n_enc_layers <= 0:
            raise ValueError("n_enc_layers must be a positive int, got %r" % (self.n_enc_layers,))
        if not isinstance(self.n_dec_layers, int) or self.n_dec_layers <= 0:
            raise ValueError("n_dec_layers must be a positive int, got %r" % (self.n_dec_layers,))
        if not isinstance(self.nhead, int) or self.nhead <= 0:
            raise ValueError("nhead must be a positive int, got %r" % (self.nhead,))
        if self.d_model % self.nhead != 0:
            raise ValueError(
                "d_model (%d) must be divisible by nhead (%d)" % (self.d_model, self.nhead)
            )
        if not isinstance(self.dim_ff, int) or self.dim_ff <= 0:
            raise ValueError("dim_ff must be a positive int, got %r" % (self.dim_ff,))
        if not isinstance(self.max_seq_len, int) or self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive int, got %r" % (self.max_seq_len,))
        if not isinstance(self.dropout, float) or not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be a float in [0, 1), got %r" % (self.dropout,))
        if not isinstance(self.pad_token_id, int) or self.pad_token_id < 0:
            raise ValueError(
                "pad_token_id must be a non-negative int, got %r" % (self.pad_token_id,)
            )
        if not isinstance(self.layer_norm_eps, float) or self.layer_norm_eps <= 0.0:
            raise ValueError(
                "layer_norm_eps must be a positive float, got %r" % (self.layer_norm_eps,)
            )
