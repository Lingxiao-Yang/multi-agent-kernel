import pytest


@pytest.mark.gpu
@pytest.mark.upstream
def test_hand_layernorm_preserves_the_three_normalized_dimensions():
    from kernel_study.public_preflight import hand_forward
    import torch
    x = torch.randn(3,2,4,8,device='cuda')
    w,b = (torch.randn_like(x[0]) for _ in range(2))
    result = hand_forward(40,[x,w,b])()
    torch.testing.assert_close(result,torch.nn.functional.layer_norm(x,x.shape[1:],w,b))
