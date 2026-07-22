from pathlib import Path

import pytest

import openpilot.common.file_chunker as file_chunker


def test_valid_chunk_set_round_trip_with_padding(tmp_path, monkeypatch):
  monkeypatch.setattr(file_chunker, "CHUNK_SIZE", 4)
  model = tmp_path / "model.pkl"
  model.write_bytes(b"abcdef")
  targets = file_chunker.get_chunk_targets(model, 12)

  file_chunker.chunk_file(model, targets)

  assert file_chunker.is_file_chunked_valid(model, require_manifest=True)
  assert file_chunker.get_existing_chunks(model) == targets
  with file_chunker.open_file_chunked(model) as f:
    assert f.read() == b"abcdef"


def test_missing_chunk_invalidates_manifest(tmp_path, monkeypatch):
  monkeypatch.setattr(file_chunker, "CHUNK_SIZE", 4)
  model = tmp_path / "model.pkl"
  model.write_bytes(b"abcdef")
  targets = file_chunker.get_chunk_targets(model, 8)
  file_chunker.chunk_file(model, targets)
  Path(targets[-1]).unlink()

  assert not file_chunker.is_file_chunked_valid(model, require_manifest=True)
  with pytest.raises(ValueError, match="incomplete chunk set"):
    file_chunker.open_file_chunked(model)


def test_nonzero_chunk_after_partial_chunk_is_invalid(tmp_path, monkeypatch):
  monkeypatch.setattr(file_chunker, "CHUNK_SIZE", 4)
  model = tmp_path / "model.pkl"
  Path(file_chunker.get_manifest_path(model)).write_text("3")
  Path(file_chunker.get_chunk_name(model, 0, 3)).write_bytes(b"ab")
  Path(file_chunker.get_chunk_name(model, 1, 3)).write_bytes(b"cdef")
  Path(file_chunker.get_chunk_name(model, 2, 3)).write_bytes(b"")

  assert not file_chunker.is_file_chunked_valid(model, require_manifest=True)
  with pytest.raises(ValueError, match="invalid chunk size"):
    file_chunker.get_existing_chunks(model)


@pytest.mark.parametrize("manifest", ["", "nope", "0", "-1"])
def test_invalid_manifest_is_rejected(tmp_path, manifest):
  model = tmp_path / "model.pkl"
  Path(file_chunker.get_manifest_path(model)).write_text(manifest)

  assert not file_chunker.is_file_chunked_valid(model, require_manifest=True)


def test_cleanup_stale_temps_is_scoped_to_requested_output(tmp_path):
  model = tmp_path / "model.pkl"
  stale_paths = [
    tmp_path / ".model.pkl.dead",
    tmp_path / "model.pkl.chunkmanifest.tmp.123",
    tmp_path / "model.pkl.chunk01of02.tmp.123",
  ]
  preserved_paths = [
    tmp_path / "model.pkl.chunkmanifest",
    tmp_path / "model.pkl.chunk01of01",
    tmp_path / "model.pkl.backup",
    tmp_path / ".other.pkl.dead",
  ]
  for path in [*stale_paths, *preserved_paths]:
    path.write_bytes(b"data")

  removed = file_chunker.cleanup_stale_file_chunk_temps(model)

  assert set(removed) == set(stale_paths)
  assert all(not path.exists() for path in stale_paths)
  assert all(path.exists() for path in preserved_paths)


def test_cleanup_incomplete_chunks_without_manifest(tmp_path):
  model = tmp_path / "model.pkl"
  incomplete = [
    model,
    Path(file_chunker.get_chunk_name(model, 0, 2)),
    Path(file_chunker.get_chunk_name(model, 1, 2)),
  ]
  unrelated = tmp_path / "model.pkl.backup"
  for path in [*incomplete, unrelated]:
    path.write_bytes(b"data")

  removed = file_chunker.cleanup_incomplete_file_chunks(model)

  assert set(removed) == set(incomplete)
  assert all(not path.exists() for path in incomplete)
  assert unrelated.exists()


def test_cleanup_invalid_manifest_and_published_chunks(tmp_path):
  model = tmp_path / "model.pkl"
  manifest = Path(file_chunker.get_manifest_path(model))
  chunk = Path(file_chunker.get_chunk_name(model, 0, 1))
  model.write_bytes(b"partial")
  manifest.write_text("invalid")
  chunk.write_bytes(b"published")

  removed = file_chunker.cleanup_incomplete_file_chunks(model)

  assert set(removed) == {model, manifest, chunk}
  assert all(not path.exists() for path in (model, manifest, chunk))


def test_cleanup_preserves_valid_manifest_set(tmp_path, monkeypatch):
  monkeypatch.setattr(file_chunker, "CHUNK_SIZE", 4)
  model = tmp_path / "model.pkl"
  manifest = Path(file_chunker.get_manifest_path(model))
  valid_chunks = [Path(file_chunker.get_chunk_name(model, i, 2)) for i in range(2)]
  orphan = Path(file_chunker.get_chunk_name(model, 0, 3))
  model.write_bytes(b"stale monolithic output")
  manifest.write_text("2")
  valid_chunks[0].write_bytes(b"abcd")
  valid_chunks[1].write_bytes(b"ef")
  orphan.write_bytes(b"orphan")

  removed = file_chunker.cleanup_incomplete_file_chunks(model)

  assert set(removed) == {model, orphan}
  assert manifest.exists()
  assert all(path.exists() for path in valid_chunks)
  assert file_chunker.is_file_chunked_valid(model, require_manifest=True)


def test_chunk_stream_pins_generation_across_replacements(tmp_path):
  chunks = [tmp_path / "chunk1", tmp_path / "chunk2"]
  chunks[0].write_bytes(b"old1")
  chunks[1].write_bytes(b"old2")
  stream = file_chunker.ChunkStream(chunks)
  try:
    first = bytearray(4)
    assert stream.readinto(first) == 4
    assert first == b"old1"

    for index, chunk in enumerate(chunks, start=1):
      replacement = tmp_path / f"replacement{index}"
      replacement.write_bytes(f"new{index}".encode())
      replacement.replace(chunk)

    second = bytearray(4)
    assert stream.readinto(second) == 4
    assert second == b"old2"
  finally:
    stream.close()


def test_chunk_stream_closes_open_files_when_later_open_fails(tmp_path, monkeypatch):
  first = tmp_path / "first"
  first.write_bytes(b"data")
  opened = []
  real_open = open

  def tracked_open(*args, **kwargs):
    f = real_open(*args, **kwargs)
    opened.append(f)
    return f

  monkeypatch.setattr(file_chunker, "open", tracked_open, raising=False)
  with pytest.raises(FileNotFoundError):
    file_chunker.ChunkStream([first, tmp_path / "missing"])

  assert len(opened) == 1
  assert opened[0].closed


def test_open_file_chunked_rejects_manifest_replaced_while_opening(tmp_path, monkeypatch):
  model = tmp_path / "model.pkl"
  manifest = Path(file_chunker.get_manifest_path(model))
  chunk = Path(file_chunker.get_chunk_name(model, 0, 1))
  manifest.write_text("1")
  chunk.write_bytes(b"old")
  real_stream = file_chunker.ChunkStream

  def replace_manifest(paths):
    stream = real_stream(paths)
    replacement = tmp_path / "replacement.manifest"
    replacement.write_text("1")
    replacement.replace(manifest)
    return stream

  monkeypatch.setattr(file_chunker, "ChunkStream", replace_manifest)
  with pytest.raises(RuntimeError, match="chunk manifest changed"):
    file_chunker.open_file_chunked(model)
