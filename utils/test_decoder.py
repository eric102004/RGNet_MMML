import torch
from RGNet_MMML.rgnet.mamba_decoder import MambaDecoderWrapper, ModelArgs, MambaDecoder

def test_mamba_decoder():
    B, M, T, D = 2, 10, 5, 32
    args = ModelArgs(d_model=D, n_layer=4, vocab_size=1)
    decoder = MambaDecoder(
        args=args,
        num_layers=args.n_layer,
        max_mem_len=M,
        max_tgt_len=T,
        use_pos_emb=True,
        final_norm=True,
        return_intermediate=False
    )
    memory = torch.randn(B, M, D)
    tgt    = torch.randn(B, T, D)
    out = decoder(memory, tgt)
    print("MambaDecoder output shape:", out.shape)
    assert out.shape == (B, T, D)


def test_mamba_decoder_wrapper():
    B, M, T, D = 2, 10, 5, 32
    num_layers = 4
    args = ModelArgs(d_model=D, n_layer=num_layers, vocab_size=1)
    wrapper = MambaDecoderWrapper(
        args=args,
        num_layers=num_layers,
        num_queries=T,
        max_src_len=M,
        max_tgt_len=T,
        use_pos_emb=True,
        final_norm=True
    )
    # build DETR‑style inputs
    memory = torch.randn(M, B, D)
    tgt    = torch.zeros(T, B, D)
    pos_embed  = torch.randn(M, B, D)
    query_pos  = torch.randn(T, B, D)
    hs = wrapper(
        tgt=tgt,
        memory=memory,
        memory_key_padding_mask=None,
        pos=pos_embed,
        query_pos=query_pos
    )
    print("MambaWrapper output shape:", hs.shape)
    # should be (num_layers, T, B, D)
    assert hs.shape == (num_layers, T, B, D)


if __name__ == "__main__":
    test_mamba_decoder()
    test_mamba_decoder_wrapper()
    print("All tests passed!")