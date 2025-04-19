import torch
import torch.nn as nn
from rgnet.mamba_minimal import ModelArgs, ResidualBlock, RMSNorm

class MambaDecoderWrapper(nn.Module):
    """
    Wraps MambaDecoder to match DETR’s TransformerDecoder API:
      forward(
        tgt:                    (Q, B, D),
        memory:                 (S, B, D),
        tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask,
        pos: (S, B, D), query_pos: (Q, B, D)
      ) -> (num_layers, Q, B, D)
    """
    def __init__(
        self,
        args: ModelArgs,
        num_layers:    int,
        num_queries:   int,
        max_src_len:   int,
        max_tgt_len:   int,
        use_pos_emb:   bool = True,
        final_norm:    bool = True
    ):
        super().__init__()
        self.num_queries = num_queries
        self.mamba = MambaDecoder(
            args=args,
            num_layers=num_layers,
            max_mem_len=max_src_len,
            max_tgt_len=max_tgt_len,
            use_pos_emb=use_pos_emb,
            final_norm=final_norm,
            return_intermediate=True
        )

    def forward(
        self,
        tgt:                        torch.Tensor,   # (Q, B, D)
        memory:                     torch.Tensor,   # (S, B, D)
        memory_key_padding_mask:    torch.Tensor = None,
        pos:                        torch.Tensor = None, # (S, B, D)
        query_pos:                  torch.Tensor = None  # (Q, B, D)
    ) -> torch.Tensor:
        Q, B, D = tgt.shape
        S, _, _ = memory.shape

        # 1) build batch‑first tensors
        mem_b = memory.permute(1, 0, 2).clone()             # (B, S, D)
        tgt_b = torch.zeros(B, Q, D, device=mem_b.device)   # (B, Q, D)

        # 2) zero‑out padded memory if mask given
        if memory_key_padding_mask is not None:
            # (B, S)
            mem_b = mem_b.masked_fill(
                memory_key_padding_mask.unsqueeze(-1),
                0.0
            )

        # 3) add DETR pos/query_pos
        if pos is not None:
            mem_b = mem_b + pos.permute(1, 0, 2)
        if query_pos is not None:
            tgt_b = tgt_b + query_pos.permute(1, 0, 2)

        # 4) run MambaDecoder
        #    → (num_layers, B, S+Q, D)
        inter = self.mamba(mem_b, tgt_b)

        # 5) slice off last Q positions → (num_layers, B, Q, D)
        inter_q = inter[:, :, -Q:, :]

        # 6) permute → (num_layers, Q, B, D)
        return inter_q.permute(0, 2, 1, 3)

class MambaDecoder(nn.Module):
    """
    A purely‑SSM seq2seq “decoder” that never uses attention:
      1) Optionally add positional embeddings to encoder & decoder inputs.
      2) Concatenate [memory, tgt] along the time axis.
      3) Run the entire (mem_len + tgt_len) sequence through N Mamba SSM ResidualBlocks.
         - Each block: Norm → SSM → Residual add.
      4) Final RMSNorm.
      5) Slice off only the last tgt_len positions to produce decoder output.
    """

    def __init__(
        self,
        args: ModelArgs,
        num_layers: int,
        max_mem_len: int,
        max_tgt_len: int,
        use_pos_emb: bool = True,
        final_norm: bool = True,
        return_intermediate: bool = False
    ):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(num_layers)])
        self.final_norm = RMSNorm(args.d_model) if final_norm else nn.Identity()
        self.return_intermediate = return_intermediate

        if use_pos_emb:
            # simple learned absolute positional encoding
            total_len = max_mem_len + max_tgt_len
            self.pos_emb = nn.Embedding(total_len, args.d_model)
        else:
            self.pos_emb = None

    def forward(
        self,
        memory_emb: torch.Tensor,   # (batch, mem_len, d_model)
        tgt_emb: torch.Tensor        # (batch, tgt_len, d_model)
    ) -> torch.Tensor:
        b, mem_len, d = memory_emb.shape
        _, tgt_len, _ = tgt_emb.shape

        # 1) add positional embeddings, if using
        if self.pos_emb is not None:
            # build position indices [0..mem_len+tgt_len)
            positions = torch.arange(mem_len + tgt_len, device=memory_emb.device)
            pos = self.pos_emb(positions)          # (mem_len+tgt_len, d_model)
            pos = pos.unsqueeze(0).expand(b, -1, -1)  # (batch, mem_len+tgt_len, d_model)

            memory_emb = memory_emb + pos[:, :mem_len]
            tgt_emb    = tgt_emb    + pos[:, mem_len:]

        # 2) concatenate encoder memory + decoder input
        print("0")
        x = torch.cat([memory_emb, tgt_emb], dim=1)  # (batch, mem_len+tgt_len, d_model)

        # 3) run through N Mamba ResidualBlocks
        intermediates = []
        for block in self.layers:
            x = block(x)  # each block does: x = Norm(x); y = SSM(x); return x+y
            if self.return_intermediate:
                # store after final norm but before next block
                intermediates.append(self.final_norm(x))

        print("1")
        # 4) final normalization
        x = self.final_norm(x)
        if self.return_intermediate:
            intermediates[-1] = x
            # stack → (num_layers, batch, total_len, d_model)
            return torch.stack(intermediates, dim=0)
        print("2")
        # 5) slice off just the decoder part
        out = x[:, -tgt_len:, :]  # (batch, tgt_len, d_model)
        return out
