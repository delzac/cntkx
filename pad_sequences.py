import cntk as C
import numpy as np


a = C.sequence.input_variable(1)
b = C.sequence.unpack(a, padding_value=0, no_mask_output=True)
c = C.splice(C.Constant(0), a, axis=0)

n = np.random.random((5, 1))
print(n)
print(c.eval({a: [n]}))
