"""Bound corrected RGB history while retaining immutable sinks and chronological motion anchors."""
import torch


def corrected_history(previous, previous_times, encoded22, head, sinks, mid_count):
    if sinks < 1 or mid_count < 0 or encoded22.shape[2] != 7:
        raise ValueError('invalid corrected-history geometry')
    if previous.shape[2] != len(previous_times) or previous.shape[2] < sinks + 2:
        raise ValueError('history and temporal coordinates disagree')
    # Native22 positions [0,1,5,9,13,17,18] start at head-21. The first
    # two overlap old context; slots 2:5 supply three new motion anchors.
    middle_times = previous_times.new_tensor([head-16, head-12, head-8])
    indices = torch.arange(sinks, previous.shape[2]-1, device=previous_times.device)
    indices = indices[previous_times[indices] < middle_times[0]]
    middle = torch.cat((previous.index_select(2, indices.to(previous.device)), encoded22[:, :, 2:5]), 2)
    times = torch.cat((previous_times[indices], middle_times))
    if mid_count:
        middle, times = middle[:, :, -mid_count:], times[-mid_count:]
    else:
        middle, times = middle[:, :, :0], times[:0]
    history = torch.cat((previous[:, :, :sinks], middle, encoded22[:, :, -2:]), 2)
    clock = torch.cat((previous_times[:sinks], times, previous_times.new_tensor([head-4, head-3])))
    if not torch.all(clock[1:] > clock[:-1]) or history.shape[2] > sinks + mid_count + 2:
        raise RuntimeError('corrected history is not chronological and bounded')
    return history.detach().contiguous(), clock
