from __future__ import annotations

import diagnostics
import harness
import modelkit
import prompting
import support
import toolkit


def test_sibling_package_root_exports_are_sorted_and_resolve() -> None:
    for package in (diagnostics, harness, modelkit, prompting, support, toolkit):
        assert list(package.__all__) == sorted(package.__all__)
        for name in package.__all__:
            assert hasattr(package, name), name
