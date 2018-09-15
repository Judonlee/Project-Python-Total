from collections import Counter
from enum import Enum

class FeatureSets(Enum):
    spectrogram = 0
    egemaps = 1
    compare = 2


def get_class_weights(y, smooth_factor=0):
    """Get the proper weights for per class

    Args:
        y: list of true labels (the labels must be hashable)
        smooth_factor: factor that smooths extremely uneven weights

    Returns:
        A dict for the weights for each class based on the frequencies of the samples
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())
    return {cls: float(majority) / count for cls, count in counter.items()}
