import torch

from openfold_local.model.primitives import Attention


def test_torch_attention_combines_additive_biases():
    attention = Attention(
        c_q=4,
        c_k=4,
        c_v=4,
        c_hidden=4,
        no_heads=1,
        gating=False,
    )
    with torch.no_grad():
        for projection in (
            attention.linear_q,
            attention.linear_k,
            attention.linear_v,
            attention.linear_o,
        ):
            projection.weight.copy_(torch.eye(4))

    q_x = torch.randn(1, 3, 4)
    kv_x = torch.randn(1, 3, 4)
    biases = [torch.randn(1, 1, 3, 3), torch.randn(1, 1, 3, 3)]

    q, k, v = attention._prep_qkv(q_x, kv_x)
    expected = torch.softmax(q @ k.transpose(-2, -1) + sum(biases), dim=-1) @ v
    expected = attention._wrap_up(expected.transpose(-2, -3), q_x)

    actual = attention(q_x, kv_x, biases=biases, triangle_attention="torch")

    torch.testing.assert_close(actual, expected)
