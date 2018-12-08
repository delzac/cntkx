import cntk as C
from cntkx.ops import cumsum
import numpy as np


def test_cumsum():
    a = C.input_variable(5)
    b = cumsum(a)

    n = np.array([1, 2, 3, 4, 5]).astype(np.float32)[None, ...]
    results = b.eval({a: n})
    np.testing.assert_equal(results[0], n.cumsum())
