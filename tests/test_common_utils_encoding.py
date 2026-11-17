"""Local notes and state files are UTF-8, not the host locale codec.

``secure_write_text`` is what diagnostic notes, the daemon descriptor and the
portable cookie file go through. Readers in this server open those files as
UTF-8, so a write in the locale codec either raises inside the caller or leaves
bytes no UTF-8 reader can decode.

stickerdaniel/linkedin-mcp-server#1511.
"""

from linkedin_mcp_server.common_utils import secure_write_text


def test_secure_write_text_writes_utf8(tmp_path):
    content = "- Error: profile \U0001f680 \u4e2d\u6587 unavailable\n"
    path = tmp_path / "note.md"

    secure_write_text(path, content)

    assert path.read_bytes() == content.encode("utf-8")
    assert path.read_text(encoding="utf-8") == content
