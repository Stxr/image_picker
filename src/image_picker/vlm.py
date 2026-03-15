from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommentRequest:
    image_path: Path
    final_score: float
    bucket: str


class NullCommentGenerator:
    """Placeholder interface for future local multimodal review."""

    enabled = False

    def generate(self, request: CommentRequest) -> str:
        return ""


def build_comment_generator(enabled: bool) -> NullCommentGenerator:
    _ = enabled
    return NullCommentGenerator()
