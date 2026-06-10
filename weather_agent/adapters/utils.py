from __future__ import annotations

import math


def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int, float, float]:
    n = 2 ** z
    x_float = (lon + 180.0) / 360.0 * n
    y_float = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x_float), int(y_float), x_float, y_float


def wind_uv(speed: float, direction_deg: float) -> tuple[float, float]:
    # meteorological direction (from) -> math vector (to)
    rad = math.radians(direction_deg)
    u = -speed * math.sin(rad)
    v = -speed * math.cos(rad)
    return u, v


def shear_from_winds(speed_low: float, dir_low: float, speed_high: float, dir_high: float) -> float:
    u1, v1 = wind_uv(speed_low, dir_low)
    u2, v2 = wind_uv(speed_high, dir_high)
    du = u2 - u1
    dv = v2 - v1
    return (du ** 2 + dv ** 2) ** 0.5


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
