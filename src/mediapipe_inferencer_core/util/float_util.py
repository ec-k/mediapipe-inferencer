def clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max_value, max(value, min_value))

def lerp(from_value: float, to_value: float, lerp_amount: float) -> float:
    amount = clamp(lerp_amount, 0, 1)
    return amount * to_value + (1-amount) * from_value