from app.autoscaler import Autoscaler, LoadSample
from app.config import Settings


def _settings():
    return Settings(
        min_replicas=1,
        max_replicas=10,
        target_concurrency_per_replica=4.0,
        scale_up_cooldown_s=0.0,
        scale_down_cooldown_s=60.0,
    )


def test_scales_up_with_load():
    clock = [0.0]
    a = Autoscaler(_settings(), clock=lambda: clock[0])
    d = a.step(LoadSample(in_flight=8, queue_depth=8))  # 16 / 4 = 4
    assert d.desired_replicas == 4
    assert a.current_replicas == 4


def test_respects_max_replicas():
    clock = [0.0]
    a = Autoscaler(_settings(), clock=lambda: clock[0])
    d = a.step(LoadSample(in_flight=100, queue_depth=100))
    assert d.desired_replicas == 10  # capped


def test_never_below_min():
    clock = [0.0]
    a = Autoscaler(_settings(), clock=lambda: clock[0])
    d = a.step(LoadSample(in_flight=0, queue_depth=0))
    assert d.desired_replicas == 1


def test_scale_down_suppressed_by_cooldown():
    clock = [0.0]
    a = Autoscaler(_settings(), clock=lambda: clock[0])
    a.step(LoadSample(in_flight=20, queue_depth=0))  # scale up to 5
    assert a.current_replicas == 5
    # Immediately drop load; cooldown should suppress the scale-down.
    d = a.step(LoadSample(in_flight=0, queue_depth=0))
    assert not d.changed
    assert a.current_replicas == 5
    # After cooldown elapses, it scales down.
    clock[0] += 61.0
    d = a.step(LoadSample(in_flight=0, queue_depth=0))
    assert d.desired_replicas == 1


def test_scale_up_is_immediate():
    clock = [0.0]
    a = Autoscaler(_settings(), clock=lambda: clock[0])
    a.step(LoadSample(in_flight=4, queue_depth=0))  # 1 replica
    d = a.step(LoadSample(in_flight=40, queue_depth=0))  # jump
    assert d.changed
    assert d.desired_replicas == 10
