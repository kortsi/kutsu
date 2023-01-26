"""Util"""
from __future__ import annotations

import hashlib
from typing import Any, cast

import pygments.lexers
import pygments.util
import rich.console
import rich.syntax


def bytes_to_readable(num_bytes: float) -> str:
    """Convert bytes to a human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f'{num_bytes:.2f} {unit}'
        num_bytes /= 1024.0
    return f'{num_bytes:.2f} {unit}'


def make_object_id(obj: Any, length: int = 7) -> str:
    """Make a unique object ID of given length consisting of hex digits"""
    return hashlib.sha256(str(hash(obj)).encode('utf-8')).hexdigest()[:length]


def get_lexer_for_content_type(content_type: str | None) -> str:
    if content_type is not None:
        mime_type, _, _ = content_type.partition(';')
        try:
            return cast(
                str,
                pygments.lexers.get_lexer_for_mimetype(mime_type.strip()
                                                       ).name  # type: ignore
            )
        except pygments.util.ClassNotFound:
            pass
    return ""


def get_console(**kwargs: Any) -> rich.console.Console:
    return rich.console.Console(**kwargs)
