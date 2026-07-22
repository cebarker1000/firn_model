"""Time-lstepping utilities."""


def run_model(stepper, state, time, dt, final_time, after_step=None):
    """Advance to final_time and optionally call after_step(step, time, state)."""
    final_time = float(final_time)
    step = 0

    while float(time) < final_time - 1.0e-12:
        dt_now = min(float(dt), final_time - float(time))
        dt.assign(dt_now)
        stepper.advance()
        time.assign(float(time) + dt_now)
        step += 1

        if after_step is not None:
            after_step(step, float(time), state)

    return step