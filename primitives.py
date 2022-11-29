from operations import OPS

PRIMITIVES_tiny = [
    'skip_connect',
    'nor_conv_1x1',
    'nor_conv_3x3',
]

CONFIG = {
    'primitives': PRIMITIVES_tiny,
}


def set_primitives(search_space):
    if isinstance(search_space, list):
        for k in search_space:
            if not k in OPS:
                raise ValueError("Not supported operation: %s" % k)
        CONFIG['primitives'] = search_space
    elif search_space == 'tiny':
        CONFIG['primitives'] = PRIMITIVES_tiny
    else:
        raise ValueError("No search space %s" % search_space)


def get_primitives():
    return CONFIG['primitives']