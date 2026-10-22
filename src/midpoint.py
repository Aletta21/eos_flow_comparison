def second_order_midpoint(
    w_prev,
    w,
    w_next,
):
    """
    Official Central Flows second-order midpoint:

        w*_t = (w_{t-1} + 2 w_t + w_{t+1}) / 4
    """
    return (
        w_prev
        + 2.0 * w
        + w_next
    ) / 4.0