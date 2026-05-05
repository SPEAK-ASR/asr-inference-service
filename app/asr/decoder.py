"""Incremental decoder.

Implements the partial/final emission policy from section 7.2 of
`investigated_detail.md`:

- A partial event is emitted only when the decoded text changes meaningfully
  (>= ``min_partial_char_delta`` characters different from the last emitted
  partial).
- ``is_stable=True`` is reported when the longest common prefix between the
  most recent two hypotheses has grown — this prefix is what the UI can
  confidently render without flicker.
- A final event is generated once on `end_utterance` / `stop`, and the
  decoder is reset for the next utterance.

The decoder is purely textual; it doesn't talk to the model directly. The
streaming engine produces hypothesis text, the decoder decides what (if
anything) to publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _longest_common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


@dataclass
class DecoderOutput:
    text: str
    is_stable: bool


@dataclass
class IncrementalDecoder:
    """Per-utterance state used to decide what partials/finals to publish."""

    min_partial_char_delta: int = 1
    last_emitted_partial: str = ""
    last_hypothesis: str = ""
    stable_prefix: str = ""
    finalized_text: str = ""
    seq: int = field(default=0)

    def observe_hypothesis(self, hypothesis: str) -> DecoderOutput | None:
        """Feed a new partial hypothesis. Returns what to emit, or None."""
        hypothesis = (hypothesis or "").strip()
        if not hypothesis:
            self.last_hypothesis = ""
            return None

        new_stable = _longest_common_prefix(self.last_hypothesis, hypothesis)
        self.last_hypothesis = hypothesis

        if len(new_stable) > len(self.stable_prefix):
            self.stable_prefix = new_stable

        if abs(len(hypothesis) - len(self.last_emitted_partial)) < self.min_partial_char_delta \
                and hypothesis == self.last_emitted_partial:
            return None

        is_stable = bool(self.stable_prefix) and hypothesis.startswith(self.stable_prefix)
        self.last_emitted_partial = hypothesis
        self.seq += 1
        return DecoderOutput(text=hypothesis, is_stable=is_stable)

    def finalize(self, final_text: str) -> str:
        """Mark the current utterance final and reset internal partial state."""
        final_text = (final_text or self.last_hypothesis).strip()
        self.finalized_text = final_text
        self.last_emitted_partial = ""
        self.last_hypothesis = ""
        self.stable_prefix = ""
        self.seq = 0
        return final_text
