#!/usr/bin/env python3
import io
import sys
import math
import os
from pathlib import Path

CHUNK_SIZE = 45 * 1024 * 1024  # 45MB, under GitHub's 50MB limit

def get_chunk_name(name, idx, num_chunks):
  return f"{name}.chunk{idx+1:02d}of{num_chunks:02d}"

def get_manifest_path(name):
  return f"{name}.chunkmanifest"

def _chunk_paths(path, num_chunks):
  return [get_manifest_path(path)] + [get_chunk_name(path, i, num_chunks) for i in range(num_chunks)]


def _validated_manifest_chunks(path):
  manifest = get_manifest_path(path)
  try:
    num_chunks = int(Path(manifest).read_text().strip())
  except (OSError, ValueError) as e:
    raise ValueError(f"invalid chunk manifest for {path}: {e}") from e
  if num_chunks <= 0:
    raise ValueError(f"invalid chunk count {num_chunks} for {path}")

  chunk_paths = [get_chunk_name(path, i, num_chunks) for i in range(num_chunks)]
  try:
    sizes = [os.path.getsize(chunk_path) for chunk_path in chunk_paths]
  except OSError as e:
    raise ValueError(f"incomplete chunk set for {path}: {e}") from e

  # chunk_file writes full chunks, then at most one partial chunk, followed by
  # zero-length padding chunks when SCons' conservative size estimate is high.
  terminal_seen = False
  for chunk_path, size in zip(chunk_paths, sizes, strict=True):
    if size < 0 or size > CHUNK_SIZE or (terminal_seen and size != 0):
      raise ValueError(f"invalid chunk size {size} for {chunk_path}")
    terminal_seen |= size < CHUNK_SIZE
  if sum(sizes) <= 0:
    raise ValueError(f"empty chunk set for {path}")
  return chunk_paths


def is_file_chunked_valid(path, require_manifest=False):
  """Return whether path has a complete, structurally valid file or chunk set."""
  try:
    if os.path.isfile(get_manifest_path(path)):
      _validated_manifest_chunks(path)
      return True
    return not require_manifest and os.path.isfile(path) and os.path.getsize(path) > 0
  except (OSError, ValueError):
    return False


def cleanup_stale_file_chunk_temps(path):
  """Remove only temporary files produced while atomically building path."""
  output = Path(path)
  patterns = (
    f".{output.name}.*",
    f"{output.name}.chunkmanifest.tmp.*",
    f"{output.name}.chunk*of*.tmp.*",
  )
  removed = []
  for pattern in patterns:
    for candidate in output.parent.glob(pattern):
      try:
        if candidate.is_file() or candidate.is_symlink():
          candidate.unlink()
          removed.append(candidate)
      except OSError:
        pass
  return removed


def _published_chunk_paths(path):
  output = Path(path)
  prefix = f"{output.name}.chunk"
  published = []
  for candidate in output.parent.glob(f"{output.name}.chunk*of*"):
    index, separator, count = candidate.name.removeprefix(prefix).partition("of")
    if separator and index.isdigit() and count.isdigit():
      published.append(candidate)
  return published


def cleanup_incomplete_file_chunks(path):
  """Remove incomplete published artifacts. Caller must hold the build lock."""
  output = Path(path)
  manifest = Path(get_manifest_path(path))
  published = set(_published_chunk_paths(path))
  try:
    valid_chunks = {Path(chunk) for chunk in _validated_manifest_chunks(path)} if manifest.is_file() else None
  except (OSError, ValueError):
    valid_chunks = None

  # A valid manifest is the commit point. Preserve it and every referenced
  # chunk, but discard a leftover monolithic output or chunks from older sets.
  candidates = [output, *(published - valid_chunks)] if valid_chunks is not None else [output, manifest, *published]
  removed = []
  for candidate in candidates:
    try:
      if candidate.is_file() or candidate.is_symlink():
        candidate.unlink()
        removed.append(candidate)
    except OSError:
      pass
  return removed

def get_chunk_targets(path, file_size):
  num_chunks = math.ceil(file_size / CHUNK_SIZE)
  return _chunk_paths(path, num_chunks)

def chunk_file(path, targets):
  manifest_path, *chunk_paths = targets
  actual_num_chunks = max(1, math.ceil(os.path.getsize(path) / CHUNK_SIZE))
  assert len(chunk_paths) >= actual_num_chunks, f"Allowed {len(chunk_paths)} chunks but needs at least {actual_num_chunks}, for path {path}"
  with open(path, 'rb') as f:
    for chunk_path in chunk_paths:
      with open(chunk_path, 'wb') as out:
        out.write(f.read(CHUNK_SIZE))
  Path(manifest_path).write_text(str(len(chunk_paths)))
  os.remove(path)

def get_existing_chunks(path):
  if os.path.isfile(path):
    if os.path.getsize(path) <= 0:
      raise ValueError(f"empty file: {path}")
    return [path]
  if os.path.isfile(manifest := get_manifest_path(path)):
    return [manifest, *_validated_manifest_chunks(path)]
  raise FileNotFoundError(path)

class ChunkStream(io.RawIOBase):
  def __init__(self, paths):
    self._files = []
    self._file = None
    self._file_index = 0
    try:
      # Open every chunk before returning so atomic replacements cannot make a
      # reader combine old and new generations between successive reads.
      for path in paths:
        self._files.append(open(path, 'rb'))
    except Exception:
      for f in self._files:
        f.close()
      self._files = []
      raise

  def readable(self):
    return True

  def readinto(self, b):
    n = 0
    while n < len(b):
      if self._file is None:
        if self._file_index >= len(self._files):
          break
        self._file = self._files[self._file_index]
        self._file_index += 1

      read = self._file.readinto(memoryview(b)[n:])
      if read:
        n += read
      else:
        self._file.close()
        self._file = None
    return n

  def close(self):
    for f in self._files:
      f.close()
    self._files = []
    self._file = None
    super().close()

def open_file_chunked(path):
  manifest_path = get_manifest_path(path)
  if os.path.isfile(manifest_path):
    manifest_stat = os.stat(manifest_path)
    paths = _validated_manifest_chunks(path)
    stream = ChunkStream(paths)
    try:
      current_stat = os.stat(manifest_path)
      manifest_identity = (manifest_stat.st_dev, manifest_stat.st_ino, manifest_stat.st_size,
                           manifest_stat.st_mtime_ns, manifest_stat.st_ctime_ns)
      current_identity = (current_stat.st_dev, current_stat.st_ino, current_stat.st_size,
                          current_stat.st_mtime_ns, current_stat.st_ctime_ns)
      if current_identity != manifest_identity:
        raise RuntimeError(f"chunk manifest changed while opening {path}")
    except Exception:
      stream.close()
      raise
  elif os.path.isfile(path):
    if os.path.getsize(path) <= 0:
      raise ValueError(f"empty file: {path}")
    stream = ChunkStream([path])
  else:
    raise FileNotFoundError(path)
  return io.BufferedReader(stream)


if __name__ == "__main__":
  path = sys.argv[1]
  chunk_paths = get_chunk_targets(path, os.path.getsize(path))
  chunk_file(path, chunk_paths)
