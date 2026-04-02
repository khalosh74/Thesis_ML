from __future__ import annotations

import pytest

from Thesis_ML.script_support.cli import fail


def test_fail_prints_stderr_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        fail("synthetic failure", exit_code=7)

    assert int(exc_info.value.code) == 7
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "synthetic failure"

