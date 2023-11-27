from values.class_operations import operation
import numpy as np

add = operation(
    lambda x, y: x+y,
    [
        lambda x, y: 1,
        lambda x, y: 1,
    ],
)
sub = operation(
    lambda x, y: x-y,
    [
        lambda x, y: 1,
        lambda x, y: -1,
    ],
)

mul = operation(
    lambda x, y: x*y,
    [
        lambda x, y: y,
        lambda x, y: x,
    ],
)
truediv = operation(
    lambda x, y: x/y,
    [
        lambda x, y: 1/y,
        lambda x, y: -x/y**2,
    ],
)
rtruediv = operation(
    lambda x, y: y/x,
    [
        lambda x, y: -y/x**2,
        lambda x, y: 1/x,
    ],
)
power =  operation(
    lambda b, e: b**e,
    [
        lambda b, e: e * b**(e-1),
        lambda b, e: b**e*np.log(b),
    ],
    lambda b, e, s_b, s_e, u_b, u_e: u_b**e,
)
