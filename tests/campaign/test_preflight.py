import pytest

from kernel_study.campaign import preflight
from kernel_study.campaign.preflight import hand_forward
from kernel_study.tasks.kernelbench import experiment


def test_check_device_selects_torchs_gpu_and_records_its_uuid(monkeypatch):
    plan = experiment()
    info = {k:plan[k] for k in ('torch','triton','cuda')}
    info.update(gpu='NVIDIA A100-PCIE-40GB',uuid='bbbb',node='v330-test')
    monkeypatch.setattr(preflight,'environment',lambda: dict(info))
    monkeypatch.setattr(preflight.subprocess,'check_output',
                        lambda *args,**kwargs: 'GPU-aaaa, 100.0\nGPU-bbbb, 570.195.03\n')
    result = preflight.check_device(plan)
    assert result['uuid']=='GPU-bbbb' and result['driver']=='570.195.03'
    with pytest.raises(ValueError):
        preflight.check_device(dict(plan,driver='other'))
    with pytest.raises(ValueError):
        preflight.check_device(dict(plan,gpu='NVIDIA A800-SXM4-80GB'))


@pytest.mark.gpu
@pytest.mark.upstream
def test_hand_layernorm_preserves_the_three_normalized_dimensions():
    import torch
    x = torch.randn(3,2,4,8,device='cuda')
    w,b = (torch.randn_like(x[0]) for _ in range(2))
    result = hand_forward(40,[x,w,b])()
    torch.testing.assert_close(result,torch.nn.functional.layer_norm(x,x.shape[1:],w,b))
