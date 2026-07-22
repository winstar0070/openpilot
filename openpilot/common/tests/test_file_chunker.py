import builtins
from pathlib import Path

from openpilot.common import file_chunker
from openpilot.common.file_chunker import get_chunk_name, get_manifest_path, open_file_chunked


def test_chunked_file_is_opened_and_read_lazily(monkeypatch, tmp_path):
  path = tmp_path / "model.pkl"
  first_chunk = Path(get_chunk_name(path, 0, 2))
  second_chunk = Path(get_chunk_name(path, 1, 2))
  first_payload = b"a" * 16384
  second_payload = b"b" * 16384
  first_chunk.write_bytes(first_payload)
  second_chunk.write_bytes(second_payload)
  Path(get_manifest_path(path)).write_text("2\n")

  real_open = builtins.open
  opened_chunks = []

  def track_open(file, *args, **kwargs):
    if str(file) in (str(first_chunk), str(second_chunk)):
      opened_chunks.append(str(file))
    return real_open(file, *args, **kwargs)

  monkeypatch.setattr(file_chunker, "open", track_open, raising=False)
  with open_file_chunked(path) as stream:
    assert opened_chunks == []
    assert stream.read(1) == b"a"
    assert opened_chunks == [str(first_chunk)]
    assert stream.read(len(first_payload)) == first_payload[1:] + b"b"
    assert opened_chunks == [str(first_chunk), str(second_chunk)]
    assert stream.read() == second_payload[1:]
