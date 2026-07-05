from __future__ import annotations

from support import messages as support_messages
from support import user_message


def test_support_messages_exports_only_message_fixtures() -> None:
    assert support_messages.__all__ == ["user_message"]
    assert user_message("hello").text == "hello"
