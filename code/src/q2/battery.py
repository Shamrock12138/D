u"""
Q2 电池 SOC 与充电模型
=======================

接口
----
soc_after_task    — 任务结束后的 SOC
charge_time_to_full — SOC → 满充所需时间


充电规则 (题目要求)
--------------------
阶段1 (SOC < 0.9): 充到 SOC=0.9 耗时 0.65*T_full
阶段2 (SOC ≥ 0.9): 从 SOC 充到满 按比例消耗 0.35*T_full

因此:
  若 SOC < 0.9:
    T_chg = (0.9 - SOC) / 0.9 * (0.65 * T_full) + 0.35 * T_full
  若 SOC ≥ 0.9:
    T_chg = (1 - SOC) / 0.1 * (0.35 * T_full)
"""


def soc_after_task(energy_kwh, energy_capacity_kwh):
    u"""计算任务结束后的电池 SOC。

    SOC = 1 - E_task / E_capacity

    Parameters
    ----------
    energy_kwh : float
        任务总能耗 (kWh)
    energy_capacity_kwh : float
        电池可用容量 (E_g^use, kWh)

    Returns
    -------
    float
        任务后 SOC (0~1)
    """
    if energy_capacity_kwh <= 0:
        return 0.0
    soc = 1.0 - energy_kwh / energy_capacity_kwh
    return max(0.0, min(1.0, soc))


def charge_time_to_full(soc, full_charge_time):
    u"""计算从给定 SOC 充到满电所需时间。

    两阶段充电:
      SOC < 0.9  →  快速充到 0.9 (线性, 耗时 0.65*T_full)
                     + 慢充到 1.0 (线性, 耗时 0.35*T_full)
      SOC ≥ 0.9  →  慢充 (线性, 耗时 0.35*T_full)

    Parameters
    ----------
    soc : float
        当前 SOC (0~1)
    full_charge_time : float
        从 0 充到满的总时间 (s)

    Returns
    -------
    float
        充电时间 (s)
    """
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





# ── 自检 ──────────────────────────────────────────────────
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