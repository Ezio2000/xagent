from __future__ import annotations

import agent_runtime


def test_public_all_is_sorted_and_resolves_exports() -> None:
    assert list(agent_runtime.__all__) == sorted(agent_runtime.__all__)
    for name in agent_runtime.__all__:
        assert hasattr(agent_runtime, name), name
