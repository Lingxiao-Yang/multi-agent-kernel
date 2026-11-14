"""Logical throughput bounds; optional externally verified hardware peaks."""


def work_and_traffic(case):
    item_bytes={'float16':2,'float32':4}[case['dtype']]
    if case['family']=='matmul':
        m,n,k=case['shape']
        return {'flops':2*m*n*k,'minimum_bytes':item_bytes*(m*k+k*n+m*n)}
    rows,n=case['shape']
    extra=2*n if case['family']=='layernorm' else n if case['family']=='elementwise' else 0
    return {'flops':None,'minimum_bytes':item_bytes*(2*rows*n+extra)}


def throughput(case, milliseconds, peak_gbps=None, peak_tflops=None):
    if milliseconds<=0:
        raise ValueError('positive latency required')
    work=work_and_traffic(case)
    gbps=work['minimum_bytes']/(milliseconds*1e6)
    tflops=work['flops']/(milliseconds*1e9) if work['flops'] else None
    return {**work,'logical_gbps':gbps,'achieved_tflops':tflops,
            'fraction_peak_bandwidth':gbps/peak_gbps if peak_gbps else None,
            'fraction_peak_compute':tflops/peak_tflops if tflops is not None and peak_tflops else None,
            'note':'Logical minimum traffic is not measured DRAM traffic. Peak fractions require verified device/dtype peaks.'}
