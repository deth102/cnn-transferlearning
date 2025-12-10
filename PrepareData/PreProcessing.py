
def normalize_signal(x, NORMALIZE):
    if NORMALIZE == "min-max":
        return (x - x.min()) / (x.max() - x.min())
    elif NORMALIZE == "mean-std":
        return (x - x.mean()) / x.std()
    else:
        return x

