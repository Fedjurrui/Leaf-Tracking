import torch
from tracking.pairing import extractMasks, findPairs, emptyDuplicates

def createMaskTimeSeries(results):
    r0 = extractMasks(results[0])
    time_series = [r0]
    for idx, val in enumerate(results[1:]):
        r1 = extractMasks(val)
        pairs = findPairs(r0,r1)
        pairs = emptyDuplicates(pairs)
        r0 = [torch.tensor(e["mask2"]) for e in pairs]
        r0 = torch.stack(r0)
        time_series.append(r0.detach().numpy())
    return time_series