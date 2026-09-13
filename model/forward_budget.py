"""Enforce a budget on actual transformer calls, including guidance branches."""
from contextlib import contextmanager


@contextmanager
def forward_budget(transformer, maximum=4):
    if not 1 <= maximum <= 4:
        raise ValueError('image/video editing permits at most four DiT forwards')
    count = {'actual_dit_forwards': 0, 'maximum_dit_forwards': maximum}

    def before_forward(*_):
        if count['actual_dit_forwards'] >= maximum:
            raise RuntimeError('editing DiT forward budget exhausted')
        count['actual_dit_forwards'] += 1

    hook = transformer.register_forward_pre_hook(before_forward, prepend=True)
    try:
        yield count
    finally:
        hook.remove()
