"""岸桥停机剩余作业预测求解器单元测试。"""

import pytest

from app.forecast import (
    Blocker,
    ForecastJob,
    InfeasibleForecast,
    solve_forecast,
)


def fj(no, stack="A", tier=1, duration=10, release=0, deadline=100,
       fixed=False, fixed_start=None, fixed_crane0=None):
    return ForecastJob(
        no=no, stack=stack, stack_tier=tier, duration=duration,
        release=release, deadline=deadline, fixed=fixed,
        fixed_start=fixed_start, fixed_crane0=fixed_crane0,
    )


def fx(no, crane0=0, start=10, duration=5, stack="X", tier=1, deadline=1000):
    return fj(no, stack=stack, tier=tier, duration=duration, release=start,
              deadline=deadline, fixed=True, fixed_start=start,
              fixed_crane0=crane0)


def test_empty_makespan_zero():
    r = solve_forecast([], 2)
    assert r.jobs == [] and r.makespan == 0


def test_fixed_job_is_pinned_to_cutoff_and_crane():
    r = solve_forecast([fx("C1", crane0=0, start=10, duration=7)], 2)
    x = r.jobs[0]
    assert (x.start_minute, x.finish_minute, x.crane_no, x.fixed) == (10, 17, 1, True)
    assert r.makespan == 17


def test_fixed_job_overlapping_shutdown_infeasible():
    with pytest.raises(InfeasibleForecast) as ei:
        solve_forecast(
            [fx("C1", crane0=0, start=10, duration=7)], 1,
            raw_blockers=[Blocker(0, 12, 14)],
        )
    assert "连续作业" in ei.value.reason


def test_fixed_job_may_abut_shutdown():
    # 固定 [10,15) 与停机 [15,20) 首尾相接，允许
    r = solve_forecast(
        [fx("C1", crane0=0, start=10, duration=5)], 1,
        raw_blockers=[Blocker(0, 15, 20)],
    )
    assert r.makespan == 15


def test_two_fixed_jobs_same_crane_overlap_infeasible():
    with pytest.raises(InfeasibleForecast):
        solve_forecast([
            fx("A", crane0=0, start=10, duration=10),
            fx("B", crane0=0, start=12, duration=2),
        ], 2)


def test_pending_job_scheduled_around_shutdown():
    # 单桥：停机 [10,15)，待开工箱时长 8、窗口 [0,100)、cutoff 已并入 release=0
    # 停机前 0+8=8<=10 可行，字典序取最早开工 0
    r = solve_forecast(
        [fj("P", duration=8, release=0, deadline=100)], 1,
        raw_blockers=[Blocker(0, 10, 15)],
    )
    assert (r.jobs[0].start_minute, r.jobs[0].finish_minute) == (0, 8)

    # release=9 时停机前放不下（9+8>10），整体推迟到停机后 [15,23)
    r2 = solve_forecast(
        [fj("P", duration=8, release=9, deadline=100)], 1,
        raw_blockers=[Blocker(0, 10, 15)],
    )
    assert (r2.jobs[0].start_minute, r2.jobs[0].finish_minute) == (15, 23)


def test_pending_job_window_between_shutdowns_infeasible():
    # 停机 [8,12) 与 [18,100)：时长 10、窗口 [0,20) ->
    # 停机前最多到 8（放不下），间隙 [12,18) 仅 6，停机后超 deadline -> 无解
    with pytest.raises(InfeasibleForecast):
        solve_forecast(
            [fj("P", duration=10, release=0, deadline=20)], 1,
            raw_blockers=[Blocker(0, 8, 12), Blocker(0, 18, 100)],
        )


def test_pending_on_different_crane_ignores_other_shutdown():
    r = solve_forecast(
        [fj("P", duration=10, release=0, deadline=100, stack="B")], 2,
        raw_blockers=[Blocker(0, 0, 50)],  # 桥1全停
    )
    assert r.jobs[0].crane_no == 2
    assert (r.jobs[0].start_minute, r.jobs[0].finish_minute) == (0, 10)


def test_shutdown_on_all_cranes_during_window_infeasible():
    with pytest.raises(InfeasibleForecast):
        solve_forecast(
            [fj("P", duration=10, release=0, deadline=20)], 2,
            raw_blockers=[Blocker(0, 0, 20), Blocker(1, 0, 20)],
        )


def test_finished_lower_box_gives_actual_finish_bound():
    # 栈 A：A1 已完工（实际第 8 分钟），A2 待开工 -> A2 不得早于 8
    r = solve_forecast(
        [fj("A2", stack="A", tier=2, duration=10)], 1,
        finished_finish={"A1": 8},
        plan_items=[("A1", "A", 1), ("A2", "A", 2)],
    )
    assert r.jobs[0].start_minute == 8 and r.jobs[0].finish_minute == 18


def test_unfinished_lower_precedence():
    # 栈 A 两层均待开工，单桥：A1[0,10) -> A2[10,20)
    r = solve_forecast(
        [fj("A1", stack="A", tier=1, duration=10),
         fj("A2", stack="A", tier=2, duration=10)], 1,
    )
    m = {x.container_no: x for x in r.jobs}
    assert m["A1"].start_minute == 0 and m["A2"].start_minute == 10
    assert r.makespan == 20


def test_fixed_lower_box_pending_upper():
    # A1 进行中 [10,15)，A2 待开工 -> A2 不早于 15
    r = solve_forecast(
        [fx("A1", crane0=0, start=10, duration=5, stack="A", tier=1),
         fj("A2", stack="A", tier=2, duration=8)], 1,
        plan_items=[("A1", "A", 1), ("A2", "A", 2)],
    )
    m = {x.container_no: x for x in r.jobs}
    assert (m["A1"].start_minute, m["A1"].finish_minute) == (10, 15)
    assert (m["A2"].start_minute, m["A2"].finish_minute) == (15, 23)


def test_pending_lower_with_finished_grandparent():
    # 三层栈：A1 已完工@5，A2/A3 待开工 -> A2>=5，A3>=A2 完工
    r = solve_forecast(
        [fj("A2", stack="A", tier=2, duration=10),
         fj("A3", stack="A", tier=3, duration=4)], 1,
        finished_finish={"A1": 5},
        plan_items=[("A1", "A", 1), ("A2", "A", 2), ("A3", "A", 3)],
    )
    m = {x.container_no: x for x in r.jobs}
    assert m["A2"].start_minute == 5
    assert m["A3"].start_minute == 15


def test_makespan_then_lex_ordering():
    # 两台桥；固定 X 桥1 [10,15)；独立待开工 P/Q 各 10 分钟，窗口 [0,..)
    # 最优 makespan=15：P 桥1 [0,10)、Q 桥2 [0,10)，X[10,15)
    r = solve_forecast(
        [fx("X", crane0=0, start=10, duration=5, stack="X", tier=1),
         fj("P", stack="P", tier=1, duration=10),
         fj("Q", stack="Q", tier=1, duration=10)], 2,
        plan_items=[("X", "X", 1), ("P", "P", 1), ("Q", "Q", 1)],
    )
    assert r.makespan == 15
    m = {x.container_no: x for x in r.jobs}
    # 字典序第一箱 P：最早 0 开工、选最小桥号 1
    assert (m["P"].start_minute, m["P"].crane_no) == (0, 1)
    assert (m["Q"].start_minute, m["Q"].crane_no) == (0, 2)


def test_pending_deadline_too_tight_after_lower_bound():
    with pytest.raises(InfeasibleForecast):
        solve_forecast(
            [fj("A2", stack="A", tier=2, duration=10, deadline=14)], 1,
            finished_finish={"A1": 8},
            plan_items=[("A1", "A", 1), ("A2", "A", 2)],
        )


def test_overlapping_shutdowns_merged():
    r = solve_forecast([], 2, raw_blockers=[
        Blocker(0, 10, 14), Blocker(0, 12, 20), Blocker(1, 5, 6),
    ])
    assert [(b.crane0, b.start, b.end) for b in r.blockers] == [(0, 10, 20), (1, 5, 6)]


def test_pending_must_start_not_before_effective_release():
    # cutoff 已由服务层并入 release；此处直接以 release=12 验证
    r = solve_forecast(
        [fj("P", duration=8, release=12, deadline=100)], 1,
    )
    assert r.jobs[0].start_minute == 12


def test_pending_job_overlapping_every_crane_at_release_infeasible():
    # release=10 单箱 10 分钟；桥1 停机 [10,18)、桥2 停机 [0,30)；
    # deadline=17 -> 桥1 最早 18 开工超 deadline，无解
    with pytest.raises(InfeasibleForecast):
        solve_forecast(
            [fj("P", duration=10, release=10, deadline=17)], 2,
            raw_blockers=[Blocker(0, 10, 18), Blocker(1, 0, 30)],
        )


def test_fixed_upper_before_finished_lower_infeasible():
    # A1 实际第 12 分钟才完工，但进行中 A2 自第 10 分钟已在作业 -> 非法
    with pytest.raises(InfeasibleForecast):
        solve_forecast(
            [fx("A2", crane0=0, start=10, duration=5, stack="A", tier=2)], 1,
            finished_finish={"A1": 12},
            plan_items=[("A1", "A", 1), ("A2", "A", 2)],
        )
