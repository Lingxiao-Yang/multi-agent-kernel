"""Run one public case and phase per GPU process before model search."""
import subprocess

from ..judge.correctness import check_forward
from ..judge.evaluate import validate
from ..judge.interface import local_forward, render_native
from ..judge.timing import calibrate, check_noise, first_call, time_pair
from ..records import digest, environment, read_json, write_json
from ..tasks import kernelbench, starts


def hand_forward(problem, inputs):
    if problem == 2:
        from tritonbench.operators.gemm.triton_matmul import matmul
        return lambda: matmul(*inputs)
    if problem == 23:
        from tritonbench.operators.softmax.operator import triton_softmax_fn
        return lambda: triton_softmax_fn(inputs[0])
    if problem == 40:
        from tritonbench.operators.layer_norm.tutorial import layer_norm
        x,w,b = inputs
        return lambda: layer_norm(x.flatten(1), (w.numel(),), w.flatten(), b.flatten(), 1e-5).view_as(x)
    raise ValueError('no matching standalone hand-written forward registered')


def check_device(plan):
    info = environment()
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version',
                                   '--format=csv,noheader'],text=True).strip().split(', ')
    if (len(gpu)!=2 or digest(gpu[0])!=plan['device_uuid_sha256'] or gpu[1]!=plan['driver']
            or any(info[k]!=plan[k] for k in ('gpu','torch','triton','cuda'))):
        raise ValueError(f'GPU or software differs from registered environment: {gpu}, {info}')
    return dict(info, uuid=gpu[0], driver=gpu[1])


def run(case, phase, directory, plan):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    seed = plan['preflight_seeds'][int(phase=='hold')]
    source, config = starts.start(case)
    path = directory/'device.py'
    path.write_text(source)
    native_path = directory/'input.py'
    native_path.write_text(kernelbench.matvec_path().read_text() if case['public_id']==4 else
                           render_native(case,source,config))
    protocol = plan['timing']
    result = dict(case=case,seed=seed,protocol=protocol,launch=config,device_source_hash=digest(source),
                  native_source_hash=digest(native_path.read_text()),problem_hash=digest(kernelbench.task_path(case).read_text()))
    if phase == 'stress':
        result['checks'] = {}
        for interface, source_path in [('native',native_path),('declarative',path)]:
            result['checks'][interface],_ = validate(case,source_path,config,interface,[seed],True)
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        return result
    model, inputs, buffers = kernelbench.task_inputs(case,seed)
    native = starts.load_native(native_path)
    forward = lambda: native(*buffers)
    local = local_forward(case,path,config,buffers)
    eager = lambda: model(*inputs)
    if phase == 'hand':
        kernelbench.upstream('tritonbench')
        hand = hand_forward(case['public_id'],buffers)
        value,seconds = first_call(hand)
        del value
        result['check'] = check_forward(lambda *args: hand(),buffers,eager(),case)
        candidate,start = time_pair(hand,forward,protocol)
        result.update(first_call_seconds=seconds,forward=candidate,start=start,
                      paired_speedup=start['median_ms']/candidate['median_ms'])
    elif phase == 'preflight':
        expected = eager()
        if list(expected.shape)!=case['output_shape']:
            raise ValueError('public task shape differs from manifest')
        compiled = torch.compile(model, mode=plan['compile_mode'],fullgraph=True)
        compiled_forward = lambda: compiled(*inputs)
        result['checks'], result['timings'] = {}, {}
        for label,fn in [('native',forward),('local',local),('compile',compiled_forward),('eager',eager)]:
            value, seconds = first_call(fn)
            del value
            result['checks'][label] = check_forward(lambda *args: fn(),buffers,expected,case)
            timed, base = time_pair(fn,forward,protocol)
            result['timings'][label] = dict(first_call_seconds=seconds,forward=timed,start=base,
                                            paired_speedup=base['median_ms']/timed['median_ms'])
        if not torch.equal(local(),forward()):
            raise AssertionError('starting interfaces returned different values')
        result['identical_outputs'] = True
    elif phase == 'calibrate':
        result.update(calibrate(forward,protocol,20))
    else:
        threshold = read_json(directory/'calibrate.json')['threshold']
        result['self'] = check_noise(forward,forward,protocol,20,threshold)
        result['interfaces'] = check_noise(local,forward,protocol,1,threshold)
        ratios = result['interfaces']['trials'][0]['ratios']
        result['interface_agreement'] = not (min(ratios)>threshold or min(1/r for r in ratios)>threshold)
    result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def run_preflight(args, parser):
    import torch
    plan = kernelbench.experiment()
    directory = args.output/args.case
    directory.mkdir(parents=True,exist_ok=True)
    report = dict(environment=check_device(plan),upstream=plan['upstream'])
    try:
        with torch.no_grad():
            report.update(run(plan['cases'][args.case],args.phase,directory,plan))
    except Exception as exc:
        write_json(directory/(args.phase+'.failed.json'),{**report,'error':str(exc)})
        raise
    write_json(directory/(args.phase+'.json'),report)
    print(args.phase,args.case,report['peak_allocated_bytes'],flush=True)
