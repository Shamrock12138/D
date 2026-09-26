






















def soc_after_task(energy_kwh, energy_capacity_kwh):
    















    if energy_capacity_kwh <= 0:
        return 0.0
    soc = 1.0 - energy_kwh / energy_capacity_kwh
    return max(0.0, min(1.0, soc))


def charge_time_to_full(soc, full_charge_time):
    


















    if soc >= 1.0:
        return 0.0

    T_full = full_charge_time
    SOC_THRESHOLD = 0.9
    FAST_FRAC = 0.65
    SLOW_FRAC = 0.35

    if soc < SOC_THRESHOLD:
        t_fast = (SOC_THRESHOLD - soc) / SOC_THRESHOLD * (FAST_FRAC * T_full)
        t_slow = SLOW_FRAC * T_full
        return t_fast + t_slow
    else:
        return (1.0 - soc) / (1.0 - SOC_THRESHOLD) * (SLOW_FRAC * T_full)






if __name__ == "__main__":
    vals = [
        (0.0, 1800, 1800.0),
        (0.5, 1800, (0.9-0.5)/0.9*0.65*1800 + 0.35*1800),
        (0.9, 1800, 0.35*1800),
        (0.95, 1800, 0.05/0.1*0.35*1800),
        (1.0, 1800, 0.0),
    ]
    print("SOC  →  T_chg")
    for soc, Tf, expected in vals:
        got = charge_time_to_full(soc, Tf)
        status = "PASS" if abs(got - expected) < 1e-9 else f"FAIL (expected {expected})"
        print(f"  {soc:.2f}  →  {got:.2f}s  {status}")