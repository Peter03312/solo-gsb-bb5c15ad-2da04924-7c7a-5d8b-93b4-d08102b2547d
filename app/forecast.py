"""岸桥停机时的剩余装船作业预测求解器。

给定某一**截至分钟** ``cutoff`` 的装船实绩快照，未完箱分两类：

* **进行中箱（fixed）**：已实际开工、尚未完工。保持其**实际岸桥**，自截至分钟
  连续作业至预计完工，即作业区间固定为
  ``[cutoff, cutoff + remaining_duration)``，不可改换岸桥、不可被停机打断；
* **待开工箱（flexible）**：尚无实际开工。在自身时间窗
  ``[release, deadline]`` 内（且不早于 cutoff）选择开工分钟与岸桥。

硬性约束：

* 每个未完箱恰好由一台岸桥作业一次（进行中箱岸桥已固定）；
* 同岸桥预计作业区间 ``[开工, 完工)`` 互不重叠（允许首尾相接）；
* 同堆栈下层箱完工后上层箱才可开工：直接下层已完工的，以其**实际完工分钟**
  为界；直接下层未完的，以其**预计完工分钟**为界；
* 任一预计作业区间不得与该岸桥的停机半开区间 ``[停机起, 停机止)`` 相交
  （允许首尾相接）；
* 待开工箱满足 ``max(release, cutoff) <= 开工`` 且 ``完工 <= deadline``。

优化目标：

1. 最小化全船预计最晚完工时间（makespan）``C``；
2. 在最优 ``C`` 下，按箱号升序使未完箱的 ``(开工分钟, 岸桥编号)`` 序列
   字典序最小（编码方式与装船时间表求解器一致：``K*开工 + 0 基岸桥序号``）。

无可行安排时抛出 :class:`InfeasibleForecast`，由服务层转为 422 且不保存
（本预测为纯计算，本就不写库）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pulp

# 单次 CBC 求解时间上限（秒）
TIME_LIMIT = 30


class InfeasibleForecast(Exception):
    """不存在满足约束的剩余作业安排。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ForecastJob:
    no: str
    stack: str
    stack_tier: int  # 箱所在堆栈层号（自底向上）
    duration: int  # 预计剩余/待工时长（正整数分钟）
    release: int  # 待开工箱的原始最早开工（服务层另并入 cutoff 与已完工下层界）
    deadline: int  # 最迟完工分钟
    fixed: bool = False  # 进行中箱：开工分钟与岸桥固定
    fixed_start: int | None = None
    fixed_crane0: int | None = None  # 0 基岸桥序号


@dataclass(frozen=True)
class Blocker:
    """某岸桥的停机半开区间 [start, end)（start < end）。"""

    crane0: int
    start: int
    end: int


@dataclass
class JobForecast:
    container_no: str
    start_minute: int
    finish_minute: int
    crane_no: int  # 岸桥编号，自 1 开始
    fixed: bool


@dataclass
class ForecastResult:
    jobs: list[JobForecast]
    makespan: int
    blockers: list[Blocker] = field(default_factory=list)


def _merge_blockers(blockers: Sequence[Blocker]) -> list[Blocker]:
    """合并同岸桥相互重叠/首尾相接的停机区间（输入允许重叠停机带）。"""
    by_crane: dict[int, list[tuple[int, int]]] = {}
    for b in blockers:
        by_crane.setdefault(b.crane0, []).append((b.start, b.end))
    merged: list[Blocker] = []
    for crane0, ivs in by_crane.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:  # 重叠或首尾相接（半开区间相接本就等价）
                cur_e = max(cur_e, e)
            else:
                merged.append(Blocker(crane0, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append(Blocker(crane0, cur_s, cur_e))
    merged.sort(key=lambda b: (b.crane0, b.start))
    return merged


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """两个半开区间是否相交（首尾相接不算重叠）。"""
    return max(a_start, b_start) < min(a_end, b_end)


def solve_forecast(
    jobs: Sequence[ForecastJob],
    crane_count: int,
    raw_blockers: Sequence[Blocker] = (),
    finished_finish: Mapping[str, int] | None = None,
    plan_items: Sequence[tuple[str, str, int]] | None = None,
) -> ForecastResult:
    """求解剩余作业预测；无可行安排时抛出 :class:`InfeasibleForecast`。

    :param jobs: 仅未完箱（进行中 fixed=True / 待开工 fixed=False）。
    :param finished_finish: 已完工箱号 -> 实际完工分钟，用于堆栈先后界。
    :param plan_items: 方案全部箱的 ``(箱号, 堆栈, 层)``，用于堆栈先后；
        缺省时只按 ``jobs`` 自身建立链（单元测试便捷入口）。
    """
    finished_finish = dict(finished_finish or {})
    if crane_count <= 0:
        raise InfeasibleForecast("岸桥数量必须为正整数")
    if not jobs:
        return ForecastResult(jobs=[], makespan=0,
                              blockers=_merge_blockers(raw_blockers))

    blockers = _merge_blockers(raw_blockers)
    by_no = {j.no: j for j in jobs}
    K = crane_count

    # ---- 堆栈邻接：配载方案保证同堆栈已占用层自底向上连续不断档 ----
    # 直接下层为已完工箱 -> 待开工箱 release 下界 / 进行中箱固定开工校验；
    # 直接下层同为未完箱 -> 预计先后边（含传递闭包）。
    chain_source = plan_items if plan_items is not None else [
        (j.no, j.stack, j.stack_tier) for j in jobs
    ]
    by_stack: dict[str, list[tuple[int, str]]] = {}
    for no, stack, tier in chain_source:
        by_stack.setdefault(stack, []).append((tier, no))

    finished_lower: dict[str, int] = {}  # 未完箱 -> 已完工直接下层的实际完工分钟
    edges: list[tuple[str, str]] = []    # (未完下层, 未完上层)
    for members in by_stack.values():
        members.sort()
        prev: object = None  # None=无（已并入传递的）下层；str=未完箱号；
        #                    tuple=('F', 实际完工分钟) 表示最近一个已完工箱
        for _tier, no in members:
            if no in finished_finish:
                prev = ("F", finished_finish[no])
                continue
            if no not in by_no:
                prev = None  # 防御性：箱状态必为三选一，不应进入
                continue
            if isinstance(prev, tuple):
                finished_lower[no] = prev[1]
            elif isinstance(prev, str):
                edges.append((prev, no))
            prev = no

    # 有效最早开工：待开工箱取 max(release, 已完工下层界)；
    # 进行中箱校验其固定开工不早于已完工下层实际完工。
    eff_release: dict[str, int] = {}
    for j in jobs:
        lb = finished_lower.get(j.no)
        if j.fixed:
            if lb is not None and j.fixed_start < lb:
                raise InfeasibleForecast(
                    f"进行中箱 {j.no} 自第 {j.fixed_start} 分钟连续作业，"
                    f"但其已完工下层箱最早第 {lb} 分钟才完工，堆栈先后无法满足"
                )
            eff_release[j.no] = j.fixed_start
        else:
            eff_release[j.no] = max(j.release, lb or 0)

    # ---- 预检 1：固定作业 vs 本桥停机；同桥两个固定作业不重叠 ----
    fixed_jobs = [j for j in jobs if j.fixed]
    for j in fixed_jobs:
        f_end = j.fixed_start + j.duration
        for b_iv in blockers:
            if b_iv.crane0 == j.fixed_crane0 and _overlaps(
                j.fixed_start, f_end, b_iv.start, b_iv.end
            ):
                raise InfeasibleForecast(
                    f"进行中箱 {j.no} 自第 {j.fixed_start} 分钟起在岸桥 "
                    f"{j.fixed_crane0 + 1} 上连续作业至第 {f_end} 分钟，"
                    f"与该桥停机区间 [{b_iv.start}, {b_iv.end}) 重叠，"
                    "无法保持连续作业"
                )
    fixed_by_crane: dict[int, list[ForecastJob]] = {}
    for j in fixed_jobs:
        fixed_by_crane.setdefault(j.fixed_crane0, []).append(j)
    for crane0, group in fixed_by_crane.items():
        group.sort(key=lambda x: x.fixed_start)
        for a, b in zip(group, group[1:]):
            if _overlaps(
                a.fixed_start, a.fixed_start + a.duration,
                b.fixed_start, b.fixed_start + b.duration,
            ):
                raise InfeasibleForecast(
                    f"岸桥 {crane0 + 1} 上进行中箱 {a.no} 与 {b.no} 的"
                    "连续作业区间相互重叠，截至时刻同一岸桥至多有一个在作业箱"
                )

    # ---- 预检 2：待开工箱时间窗容得下时长 ----
    for j in jobs:
        if j.fixed:
            continue
        if eff_release[j.no] + j.duration > j.deadline:
            raise InfeasibleForecast(
                f"待开工箱 {j.no} 的预计作业时长 {j.duration} 分钟无法落入其"
                f"有效时间窗 [{eff_release[j.no]}, {j.deadline}]"
                "（最早开工+时长 > 最迟完工）"
            )

    # ---- 预检 3：堆栈链自底向上传播最早完工（时间窗/固定作业） ----
    for members in by_stack.values():
        members.sort()
        earliest_finish: int | None = None
        for _tier, no in members:
            if no in finished_finish:
                earliest_finish = finished_finish[no]
                continue
            j = by_no.get(no)
            if j is None:
                earliest_finish = None
                continue
            if j.fixed:
                if (earliest_finish is not None
                        and j.fixed_start < earliest_finish):
                    raise InfeasibleForecast(
                        f"进行中箱 {j.no} 第 {j.fixed_start} 分钟已在作业，"
                        f"但同堆栈下层最早第 {earliest_finish} 分钟才完工"
                    )
                earliest_finish = j.fixed_start + j.duration
            else:
                start = max(eff_release[no], earliest_finish or 0)
                finish = start + j.duration
                if finish > j.deadline:
                    raise InfeasibleForecast(
                        f"待开工箱 {j.no} 受堆栈下层先后与时间窗限制，最早完工 "
                        f"{finish} 晚于最迟完工 {j.deadline}，无可行安排"
                    )
                earliest_finish = finish

    # ---- 未完箱之间堆栈先后的传递闭包（闭包对跳过同桥析取） ----
    idx = {j.no: i for i, j in enumerate(jobs)}
    n = len(jobs)
    reach = [[False] * n for _ in range(n)]
    for a, b in edges:
        if a in idx and b in idx:
            reach[idx[a]][idx[b]] = True
    for mid in range(n):
        for a in range(n):
            if not reach[a][mid]:
                continue
            for b in range(n):
                if reach[mid][b]:
                    reach[a][b] = True
    closure = {(a, b) for a in range(n) for b in range(n) if reach[a][b]}

    # big-M 须覆盖最迟完工、固定作业与停机带的最大时刻
    horizon = max(
        [j.deadline for j in jobs]
        + [j.fixed_start + j.duration for j in fixed_jobs]
        + [b.end for b in blockers]
        + [0]
    )
    big_m = horizon + 1
    blockers_by_crane: dict[int, list[Blocker]] = {}
    for b_iv in blockers:
        blockers_by_crane.setdefault(b_iv.crane0, []).append(b_iv)

    def build(
        makespan_bound: int | None = None,
        fixed: Mapping[str, tuple[int, int]] | None = None,
        lex_jobs: Sequence[str] | None = None,
    ):
        fixed = fixed or {}
        prob = pulp.LpProblem("remaining_forecast", pulp.LpMinimize)
        s = {j.no: pulp.LpVariable(f"s_{i}", lowBound=0, cat="Integer")
             for i, j in enumerate(jobs)}
        z = {(j.no, k): pulp.LpVariable(f"z_{i}_{k}", cat="Binary")
             for i, j in enumerate(jobs) for k in range(K)}
        C = pulp.LpVariable("C", lowBound=0, cat="Integer")

        if lex_jobs:
            H = big_m - 1
            base = K * (H + 1) + 1
            k_block = len(lex_jobs)
            prob += pulp.lpSum(
                (base ** (k_block - 1 - pos))
                * (K * s[cno]
                   + pulp.lpSum(k * z[cno, k] for k in range(K)))
                for pos, cno in enumerate(lex_jobs)
            ), "lex_objective"
        else:
            prob += C, "min_makespan"

        for j in jobs:
            prob += pulp.lpSum(z[j.no, k] for k in range(K)) == 1, f"onecrane_{j.no}"
            prob += s[j.no] + j.duration <= C, f"cmax_{j.no}"
            if j.fixed:
                prob += s[j.no] == j.fixed_start, f"fix_s_{j.no}"
                prob += z[j.no, j.fixed_crane0] == 1, f"fix_z_{j.no}"
            else:
                prob += s[j.no] >= eff_release[j.no], f"release_{j.no}"
                prob += s[j.no] + j.duration <= j.deadline, f"deadline_{j.no}"

        for cno, (st, k0) in fixed.items():
            prob += s[cno] == st, f"lock_s_{cno}"
            prob += z[cno, k0] == 1, f"lock_z_{cno}"

        # 堆栈先后（未完箱之间）
        for a, b in edges:
            if a in idx and b in idx:
                prob += s[a] + by_no[a].duration <= s[b], f"stack_{a}_{b}"

        # 同岸桥不重叠：无堆栈先后关系的箱对（含固定箱）
        for i in range(n):
            for jj in range(i + 1, n):
                if (i, jj) in closure or (jj, i) in closure:
                    continue
                ni, nj = jobs[i].no, jobs[jj].no
                pi, pj = jobs[i].duration, jobs[jj].duration
                fi, fj = jobs[i].fixed, jobs[jj].fixed
                if fi and fj:
                    # 预检已保证不同桥或同桥不相交，无需析取变量
                    continue
                order = pulp.LpVariable(f"b_{i}_{jj}", cat="Binary")
                if not fi and not fj:
                    # 两待开工箱：按所在桥析取（不同桥时 apart 松弛）
                    for k in range(K):
                        apart = 2 - z[ni, k] - z[nj, k]
                        prob += (
                            s[ni] + pi <= s[nj] + big_m * (1 - order)
                            + big_m * apart,
                            f"nb_{i}_{jj}_{k}_a",
                        )
                        prob += (
                            s[nj] + pj <= s[ni] + big_m * order + big_m * apart,
                            f"nb_{i}_{jj}_{k}_b",
                        )
                elif fi and not fj:
                    # ni 固定、nj 待开工：仅当 nj 选到固定桥时析取
                    k = jobs[i].fixed_crane0
                    fs = jobs[i].fixed_start
                    same = 1 - z[nj, k]
                    prob += (
                        s[nj] + pj <= fs + big_m * (1 - order) + big_m * same,
                        f"nb_f_{i}_{jj}_a",
                    )
                    prob += (
                        fs + pi <= s[nj] + big_m * order + big_m * same,
                        f"nb_f_{i}_{jj}_b",
                    )
                else:
                    # nj 固定、ni 待开工
                    k = jobs[jj].fixed_crane0
                    fs = jobs[jj].fixed_start
                    same = 1 - z[ni, k]
                    prob += (
                        s[ni] + pi <= fs + big_m * (1 - order) + big_m * same,
                        f"nb_f_{i}_{jj}_a2",
                    )
                    prob += (
                        fs + pj <= s[ni] + big_m * order + big_m * same,
                        f"nb_f_{i}_{jj}_b2",
                    )

        # 待开工箱避开各桥停机带（进行中箱由预检保证不撞停机）
        g_counter = 0
        for i, j in enumerate(jobs):
            if j.fixed:
                continue
            for k in range(K):
                for b_iv in blockers_by_crane.get(k, ()):
                    g = pulp.LpVariable(f"g_{g_counter}", cat="Binary")
                    g_counter += 1
                    same = 1 - z[j.no, k]
                    # g=1：作业在停机前结束；g=0：作业在停机后开工
                    prob += (
                        s[j.no] + j.duration <= b_iv.start
                        + big_m * (1 - g) + big_m * same,
                        f"blk_{g_counter}_a",
                    )
                    prob += (
                        b_iv.end <= s[j.no] + big_m * g + big_m * same,
                        f"blk_{g_counter}_b",
                    )

        if makespan_bound is not None:
            prob += C <= makespan_bound, "C_bound"
        return prob, s, z, C

    def _solve(prob) -> str:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT, gapRel=0)
        prob.solve(solver)
        return pulp.LpStatus[prob.status]

    prob1, s1, z1, C1 = build()
    status = _solve(prob1)
    if status == "Infeasible":
        raise InfeasibleForecast(
            "在给定停机区间、待开工时间窗、同岸桥不重叠与堆栈下层先完工的"
            "共同限制下，不存在可行的剩余作业安排"
        )
    if status != "Optimal":
        raise InfeasibleForecast(
            f"求解器在 {TIME_LIMIT}s 内未能判定可行安排（状态：{status}），"
            "请缩小规模或重试"
        )
    c_star = int(round(pulp.value(C1)))

    # 第二阶段：固定 C = c_star，按箱号升序分块求 (开工,岸桥) 字典序最优
    ordered = [j.no for j in sorted(jobs, key=lambda x: x.no)]
    H = big_m - 1
    base = K * (H + 1) + 1
    digits = max(1, len(str(base - 1)))
    chunk = max(1, 13 // digits)

    fixed: dict[str, tuple[int, int]] = {}
    last_s, last_z = s1, z1
    for start in range(0, n, chunk):
        block = ordered[start:start + chunk]
        prob, s, z, _ = build(
            makespan_bound=c_star, fixed=fixed, lex_jobs=block
        )
        bstatus = _solve(prob)
        if bstatus != "Optimal":
            raise InfeasibleForecast("字典序优化失败，请重试")
        for cno in block:
            st = int(round(pulp.value(s[cno])))
            k0 = max(range(K), key=lambda k: pulp.value(z[cno, k]) or 0)
            fixed[cno] = (st, k0)
        last_s, last_z = s, z

    forecasts: list[JobForecast] = []
    for cno in ordered:
        j = by_no[cno]
        st = int(round(pulp.value(last_s[cno])))
        k0 = max(range(K), key=lambda k: pulp.value(last_z[cno, k]) or 0)
        forecasts.append(JobForecast(
            container_no=cno,
            start_minute=st,
            finish_minute=st + j.duration,
            crane_no=k0 + 1,
            fixed=j.fixed,
        ))
    return ForecastResult(jobs=forecasts, makespan=c_star, blockers=blockers)
