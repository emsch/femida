def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def tupleit(t):
    return list(map(tupleit, t)) if isinstance(t, (tuple, list)) else t
