from alpine.gp.primitives import PrimitiveParams
import jax.numpy as jnp
import dctkit as dt
from jax import Array


def protectedDiv(left, right):
    try:
        return jnp.divide(left, right)
    except ZeroDivisionError:
        return jnp.nan


def protectedLog(x):
    try:
        return jnp.log(x)
    except ValueError:
        return jnp.nan


def protectedSqrt(x):
    try:
        return jnp.sqrt(x)
    except ValueError:
        return jnp.nan


def square_mod(x):
    return jnp.square(x).astype(dt.float_dtype)


def scalar_mul(x, k):
    return k*x


array_primitives = {
    # vector operations (JAX backend)
    'Add': PrimitiveParams(jnp.add, [Array, Array], Array),
    'Sub': PrimitiveParams(jnp.subtract, [Array, Array], Array),
    'Mul': PrimitiveParams(jnp.multiply, [Array, Array], Array),
    'Sin': PrimitiveParams(jnp.sin, [Array], Array),
    'Arcsin': PrimitiveParams(jnp.arcsin, [Array], Array),
    'Cos': PrimitiveParams(jnp.cos, [Array], Array),
    'Arccos': PrimitiveParams(jnp.arccos, [Array], Array),
    'Exp': PrimitiveParams(jnp.exp, [Array], Array),
    'Log': PrimitiveParams(protectedLog, [Array], Array),
    'Sqrt': PrimitiveParams(protectedSqrt, [Array], Array),
    'Square': PrimitiveParams(jnp.square, [Array], Array),
    'sc_mul': PrimitiveParams(scalar_mul, [Array, float], Array)
}
