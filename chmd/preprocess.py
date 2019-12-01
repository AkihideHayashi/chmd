"""Common preprocess."""
import numpy as np


def symbols_to_elements(symbols: np.ndarray,
                        order: np.ndarray) -> np.ndarray:
    """Convert symbols to uniqued elements number.

    Parameters
    ----------
    symbols         : symbols that consists molecular
    order_of_symbols: unique symbols

    Returns
    -------
    elements: where order_of_symbols[elements] == symbols

    """
    shape = symbols.shape
    s = symbols.flatten()
    condlist = s[None, :] == order[:, None]
    # assert np.all(np.any(condlist, axis=0))
    choicelist = np.arange(len(order))
    elements = np.select(condlist, choicelist).reshape(shape)
    valid = symbols != ''
    assert np.all(order[elements[valid]] == symbols[valid])
    return elements
