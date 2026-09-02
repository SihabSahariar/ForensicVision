"""Forensic integrity services: hashing, metadata, provenance and safe mode."""

from forensic.hashing import HashSet, hash_array, hash_bytes, hash_file, verify_file
from forensic.metadata import FileMetadata, extract_metadata
from forensic.provenance import ProvenanceRecord, write_sidecar
from forensic.safe_mode import SafeModeGuard, get_guard, safe_mode_enabled

__all__ = [
    "HashSet",
    "hash_file",
    "hash_bytes",
    "hash_array",
    "verify_file",
    "FileMetadata",
    "extract_metadata",
    "ProvenanceRecord",
    "write_sidecar",
    "SafeModeGuard",
    "get_guard",
    "safe_mode_enabled",
]
