"""岸桥停机剩余作业预测 API 集成测试。"""

import pytest


SLOTS = {
    "slots": [
        {"code": "A01", "stack": "A", "tier": 1, "capacity": 100, "arm": 2.0},
        {"code": "A02", "stack": "A", "tier": 2, "capacity": 100, "arm": 3.0},
        {"code": "B01", "stack": "B", "tier": 1, "capacity": 100, "arm": 2.0},
        {"code": "C01", "stack": "C", "tier": 1, "capacity": 100, "arm": 2.0},
    ],
    "adjacency": [],
    "incompatible_classes": [],
    "min_moment": -100000,
    "max_moment": 100000,
}

# A1/A2 同栈两层；B1、C1 独立底层
MANIFEST = {"containers": [
    {"no": "A1", "weight": 10, "hazmat": None, "discharge_order": 1},
    {"no": "A2", "weight": 10, "hazmat": None, "discharge_order": 1},
    {"no": "B1", "weight": 10, "hazmat": None, "discharge_order": 1},
    {"no": "C1", "weight": 10, "hazmat": None, "discharge_order": 1},
]}


@pytest.fixture
def scenario(client):
    """已确认方案 + 2 桥排程 + 实绩：A1 已完工 [0,8)；B1 桥2 进行中(start@5)；
    A2、C1 待开工。返回 (client, plan_id)。"""
    sr = client.post("/api/slot-revisions", json=SLOTS).json()["slots_revision"]
    mr = client.post("/api/manifest-revisions", json=MANIFEST).json()["manifest_revision"]
    pid = client.post("/api/plans/compute",
                      json={"slots_revision": sr, "manifest_revision": mr}).json()["plan_id"]
    client.post(f"/api/plans/{pid}/confirm",
                json={"slots_revision": sr, "manifest_revision": mr})
    client.post(f"/api/plans/{pid}/schedules", json={
        "crane_count": 2,
        "entries": [
            {"no": "A1", "duration": 8, "release_minute": 0, "deadline_minute": 100},
            {"no": "A2", "duration": 10, "release_minute": 0, "deadline_minute": 100},
            {"no": "B1", "duration": 12, "release_minute": 0, "deadline_minute": 100},
            {"no": "C1", "duration": 10, "release_minute": 0, "deadline_minute": 100},
        ],
    })
    client.post(f"/api/plans/{pid}/actual-events", json={"events": [
        {"seq": 1, "event_type": "start", "container_no": "A1", "crane_no": 1, "minute": 0},
        {"seq": 2, "event_type": "finish", "container_no": "A1", "crane_no": 1, "minute": 8},
        {"seq": 3, "event_type": "start", "container_no": "B1", "crane_no": 2, "minute": 5},
    ]})
    return client, pid


def base_body(**over):
    body = {
        "expected_version": 1,
        "cutoff_minute": 10,
        "in_progress": [{"no": "B1", "remaining_duration": 5}],
        "pending": [
            {"no": "A2", "duration": 10, "release_minute": 0, "deadline_minute": 100},
            {"no": "C1", "duration": 10, "release_minute": 0, "deadline_minute": 100},
        ],
        "shutdowns": [],
    }
    body.update(over)
    return body


def test_plan_not_found_404(client):
    r = client.post("/api/plans/nope/remaining-forecast", json=base_body())
    assert r.status_code == 404


def test_draft_plan_409(client):
    sr = client.post("/api/slot-revisions", json=SLOTS).json()["slots_revision"]
    mr = client.post("/api/manifest-revisions", json=MANIFEST).json()["manifest_revision"]
    pid = client.post("/api/plans/compute",
                      json={"slots_revision": sr, "manifest_revision": mr}).json()["plan_id"]
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=base_body())
    assert r.status_code == 409 and "确认" in r.json()["detail"]


def test_no_actuals_409(client, scenario):
    # 另起一个已确认、有排程但无实绩的方案
    client, _ = scenario
    sr = client.post("/api/slot-revisions", json=SLOTS).json()["slots_revision"]
    mr = client.post("/api/manifest-revisions", json=MANIFEST).json()["manifest_revision"]
    pid = client.post("/api/plans/compute",
                      json={"slots_revision": sr, "manifest_revision": mr}).json()["plan_id"]
    client.post(f"/api/plans/{pid}/confirm",
                json={"slots_revision": sr, "manifest_revision": mr})
    client.post(f"/api/plans/{pid}/schedules", json={
        "crane_count": 2,
        "entries": [
            {"no": n, "duration": 5, "release_minute": 0, "deadline_minute": 100}
            for n in ("A1", "A2", "B1", "C1")
        ],
    })
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json={
        "expected_version": 1, "cutoff_minute": 10,
        "in_progress": [],
        "pending": [
            {"no": n, "duration": 5, "release_minute": 0, "deadline_minute": 100}
            for n in ("A1", "A2", "B1", "C1")
        ],
        "shutdowns": [],
    })
    assert r.status_code == 409 and "实绩" in r.json()["detail"]


def test_stale_expected_version_409(client, scenario):
    client, pid = scenario
    r = client.post(f"/api/plans/{pid}/remaining-forecast",
                    json=base_body(expected_version=99))
    assert r.status_code == 409 and "版本" in r.json()["detail"]


def test_superseded_plan_409(client, scenario):
    client, pid = scenario
    # 生成修订链新修订并确认，旧修订变 superseded
    new_manifest = {"containers": [
        {"no": "A1", "weight": 11, "hazmat": None, "discharge_order": 1},
        {"no": "A2", "weight": 10, "hazmat": None, "discharge_order": 1},
        {"no": "B1", "weight": 10, "hazmat": None, "discharge_order": 1},
        {"no": "C1", "weight": 10, "hazmat": None, "discharge_order": 1},
    ]}
    mr2 = client.post("/api/manifest-revisions", json=new_manifest).json()["manifest_revision"]
    sr = client.get("/api/slot-revisions/1").json()["slots_revision"]
    p2 = client.post(f"/api/plans/{pid}/amend",
                     json={"slots_revision": sr, "manifest_revision": mr2}).json()["plan_id"]
    ok = client.post(f"/api/plans/{p2}/confirm",
                     json={"slots_revision": sr, "manifest_revision": mr2})
    assert ok.status_code == 200
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=base_body())
    assert r.status_code == 409 and "取代" in r.json()["detail"]


def test_event_after_cutoff_422(client, scenario):
    client, pid = scenario
    # 补一条第 12 分钟的事件（B1 仍在桥2，合法完工），cutoff=10 已晚
    client.post(f"/api/plans/{pid}/actual-events", json={"events": [
        {"seq": 4, "event_type": "finish", "container_no": "B1", "crane_no": 2, "minute": 12},
    ]})
    body = {
        "expected_version": 1, "cutoff_minute": 10,
        "in_progress": [],
        "pending": [
            {"no": "A2", "duration": 10, "release_minute": 0, "deadline_minute": 100},
            {"no": "C1", "duration": 10, "release_minute": 0, "deadline_minute": 100},
        ],
        "shutdowns": [],
    }
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422
    assert r.json()["events_after_cutoff"] == [4]


def test_missing_unfinished_box_422(client, scenario):
    client, pid = scenario
    body = base_body()
    body["pending"] = [body["pending"][0]]  # 丢掉 C1
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422 and "C1" in r.json()["detail"]


def test_extra_box_422(client, scenario):
    client, pid = scenario
    body = base_body()
    body["pending"].append(
        {"no": "A1", "duration": 5, "release_minute": 0, "deadline_minute": 100})
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422 and "已完工" in r.json()["detail"]


def test_status_mismatch_422(client, scenario):
    client, pid = scenario
    body = base_body()
    # B1 实绩为进行中，却放到 pending
    body["in_progress"] = []
    body["pending"].append(
        {"no": "B1", "duration": 5, "release_minute": 0, "deadline_minute": 100})
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422 and "B1" in r.json()["detail"]


def test_unknown_box_422(client, scenario):
    client, pid = scenario
    body = base_body()
    body["pending"].append(
        {"no": "ZZ9", "duration": 5, "release_minute": 0, "deadline_minute": 100})
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422


def test_shutdown_crane_out_of_range_422(client, scenario):
    client, pid = scenario
    r = client.post(f"/api/plans/{pid}/remaining-forecast",
                    json=base_body(shutdowns=[{"crane_no": 3, "start_minute": 10,
                                              "end_minute": 12}]))
    assert r.status_code == 422 and "岸桥" in r.json()["detail"]


def test_shutdown_inverted_window_422(client, scenario):
    client, pid = scenario
    r = client.post(f"/api/plans/{pid}/remaining-forecast",
                    json=base_body(shutdowns=[{"crane_no": 1, "start_minute": 12,
                                              "end_minute": 12}]))
    assert r.status_code == 422


def test_pending_window_inverted_422(client, scenario):
    client, pid = scenario
    body = base_body()
    body["pending"][0]["deadline_minute"] = 3
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422


def test_negative_or_zero_durations_422(client, scenario):
    client, pid = scenario
    body = base_body()
    body["in_progress"][0]["remaining_duration"] = 0
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422


def test_infeasible_fixed_box_hits_shutdown_422(client, scenario):
    client, pid = scenario
    # B1 桥2 [10,15)，桥2 停机 [12,14) -> 无法连续作业
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=base_body(shutdowns=[
        {"crane_no": 2, "start_minute": 12, "end_minute": 14}]))
    assert r.status_code == 422 and "连续作业" in r.json()["detail"]


def test_infeasible_pending_deadline_422(client, scenario):
    client, pid = scenario
    body = base_body()
    body["pending"][0]["deadline_minute"] = 11  # A2 最早 8 起做 10 分钟 -> 18 > 11
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422


def test_forecast_success_structure_and_precedence(client, scenario):
    client, pid = scenario
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=base_body())
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == 1
    assert data["cutoff_minute"] == 10
    assert data["crane_count"] == 2
    by = {e["container_no"]: e for e in data["entries"]}
    # B1 进行中：桥2 [10,15)
    assert (by["B1"]["start_minute"], by["B1"]["finish_minute"],
            by["B1"]["crane_no"], by["B1"]["in_progress"]) == (10, 15, 2, True)
    # A2 有效最早开工 = max(下层A1实际完工8, cutoff10) = 10
    assert by["A2"]["start_minute"] >= 10
    assert by["A2"]["in_progress"] is False
    # 时长一致
    assert by["A2"]["finish_minute"] - by["A2"]["start_minute"] == 10
    assert by["C1"]["finish_minute"] - by["C1"]["start_minute"] == 10
    assert data["makespan"] == max(e["finish_minute"] for e in data["entries"])
    # 同桥区间不重叠
    for crane in (1, 2):
        iv = sorted((e["start_minute"], e["finish_minute"])
                    for e in data["entries"] if e["crane_no"] == crane)
        for (s1, f1), (s2, f2) in zip(iv, iv[1:]):
            assert f1 <= s2


def test_forecast_avoids_shutdown_and_keeps_makespan_optimal(client, scenario):
    client, pid = scenario
    # 桥1 停机 [10,12)；B1 桥2[10,15)。
    # A2(release>=8,dur10)：桥1[8,18)撞停机->只能[12,22)，或桥2撞B1需等15->[15,25)；
    # 最优：A2 桥1 [12,22)，C1 桥2 [15,25) -> makespan 25
    r = client.post(f"/api/plans/{pid}/remaining-forecast",
                    json=base_body(shutdowns=[
                        {"crane_no": 1, "start_minute": 10, "end_minute": 12}]))
    assert r.status_code == 200
    data = r.json()
    assert data["makespan"] == 25
    by = {e["container_no"]: e for e in data["entries"]}
    assert (by["A2"]["start_minute"], by["A2"]["finish_minute"], by["A2"]["crane_no"]) == (12, 22, 1)
    assert (by["C1"]["start_minute"], by["C1"]["finish_minute"], by["C1"]["crane_no"]) == (15, 25, 2)
    # 停机区间被归一化回显
    assert data["shutdowns"] == [{"crane_no": 1, "start_minute": 10, "end_minute": 12}]
    # 无任何预计作业与停机相交
    for e in data["entries"]:
        if e["crane_no"] == 1:
            assert not (e["start_minute"] < 12 and e["finish_minute"] > 10)


def test_forecast_does_not_mutate_actuals_schedule(client, scenario):
    client, pid = scenario
    before = client.get(f"/api/plans/{pid}/actuals").json()
    sched_before = client.get(f"/api/plans/{pid}/schedules").json()
    client.post(f"/api/plans/{pid}/remaining-forecast", json=base_body())
    after = client.get(f"/api/plans/{pid}/actuals").json()
    sched_after = client.get(f"/api/plans/{pid}/schedules").json()
    assert before["version"] == after["version"] == 1
    assert before["events"] == after["events"]
    assert sched_before["entries"] == sched_after["entries"]
    assert sched_before["makespan"] == sched_after["makespan"]


def test_all_finished_empty_forecast_makespan_zero(client, scenario):
    client, pid = scenario
    # 把 A2、C1 也补完工（A2 须在 A1 完工后；用桥1，B1 仍在桥2）
    client.post(f"/api/plans/{pid}/actual-events", json={"events": [
        {"seq": 4, "event_type": "finish", "container_no": "B1", "crane_no": 2, "minute": 12},
        {"seq": 5, "event_type": "start", "container_no": "A2", "crane_no": 1, "minute": 12},
        {"seq": 6, "event_type": "finish", "container_no": "A2", "crane_no": 1, "minute": 22},
        {"seq": 7, "event_type": "start", "container_no": "C1", "crane_no": 2, "minute": 12},
        {"seq": 8, "event_type": "finish", "container_no": "C1", "crane_no": 2, "minute": 22},
    ]})
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json={
        "expected_version": 1, "cutoff_minute": 30,
        "in_progress": [], "pending": [], "shutdowns": [],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["entries"] == [] and data["makespan"] == 0


def test_cutoff_must_be_nonnegative_422(client, scenario):
    client, pid = scenario
    r = client.post(f"/api/plans/{pid}/remaining-forecast",
                    json=base_body(cutoff_minute=-1))
    assert r.status_code == 422


def test_duplicate_box_across_lists_422(client, scenario):
    client, pid = scenario
    body = {
        "expected_version": 1, "cutoff_minute": 10,
        "in_progress": [
            {"no": "B1", "remaining_duration": 5},
            {"no": "C1", "remaining_duration": 3},
        ],
        "pending": [
            {"no": "A2", "duration": 10, "release_minute": 0, "deadline_minute": 100},
            {"no": "C1", "duration": 10, "release_minute": 0, "deadline_minute": 100},
        ],
        "shutdowns": [],
    }
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=body)
    assert r.status_code == 422


def test_pending_window_before_cutoff_gets_effective_release(client, scenario):
    client, pid = scenario
    # 全部待开工箱 release=0 < cutoff=10，有效最早开工取 max(release, cutoff, 下层界)>=10
    r = client.post(f"/api/plans/{pid}/remaining-forecast", json=base_body())
    assert r.status_code == 200
    for e in r.json()["entries"]:
        if not e["in_progress"]:
            assert e["start_minute"] >= 10
