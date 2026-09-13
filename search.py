"""Layer-wise beam search over a fixed vocabulary, without I/O or model calls."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain, islice


@dataclass(frozen=True, slots=True)
class PrefixEvaluation:
    call: int
    api_choice: str
    probabilities: dict[str, float]


@dataclass(frozen=True, slots=True)
class BeamStep:
    prefix_before: str
    prefix_after: str
    label: str
    evaluation: PrefixEvaluation


@dataclass(frozen=True, slots=True)
class BeamResult:
    text: str
    completed: bool
    steps: tuple[BeamStep, ...]
    log_probability: float
    score: float | None


@dataclass(frozen=True, slots=True)
class _Node:
    prefix: str
    log_probability: float
    new_tokens: int
    parent: _Node | None = None
    step: BeamStep | None = None

    @property
    def score(self) -> float | None:
        return self.log_probability / self.new_tokens if self.new_tokens else None

    def snapshot(self) -> dict:
        return {
            "prefix": self.prefix,
            "log_probability": self.log_probability,
            "score": self.score,
            "new_tokens": self.new_tokens,
        }


class BeamSearch:
    """Prune live paths with optional token diversity; rank results by mean log score.

    Evaluations and parent-linked traces are shared, never copied per branch.
    Only a completed expansion layer is pruned, preserving interrupted work.
    """

    def __init__(
        self,
        prefix: str,
        vocabulary: dict[str, str | None],
        beam_width: int,
        max_new_chars: int,
        diversity: str = "none",
    ) -> None:
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        if isinstance(max_new_chars, bool) or not isinstance(max_new_chars, int) or max_new_chars < 0:
            raise ValueError("max_new_chars must be a nonnegative integer")
        if diversity not in ("none", "token"):
            raise ValueError("diversity must be none or token")
        if sum(token is None for token in vocabulary.values()) != 1:
            raise ValueError("vocabulary must contain exactly one STOP action")
        if any(not isinstance(label, str) or not label for label in vocabulary):
            raise ValueError("vocabulary labels must be nonempty strings")
        if any(token is not None and (not isinstance(token, str) or not token)
               for token in vocabulary.values()):
            raise ValueError("text tokens must be nonempty strings")

        self._vocabulary = dict(vocabulary)
        self._beam_width = beam_width
        self._diversity = diversity
        self._max_length = len(prefix) + max_new_chars
        self._root = _Node(prefix, 0.0, 0)
        self._frontier = [self._root]
        self._index = 0
        self._children: dict[str, _Node] = {}
        self._completed: _Node | None = None
        self._terminal_partial: _Node | None = None
        self._depth = 0

    @property
    def pending_prefix(self) -> str | None:
        if self._index < len(self._frontier):
            return self._frontier[self._index].prefix
        return None

    @property
    def depth(self) -> int:
        return self._depth

    @staticmethod
    def _better_mean(candidate: _Node, previous: _Node | None) -> bool:
        if previous is None:
            return True
        if not candidate.new_tokens:
            return False
        if not previous.new_tokens:
            return True
        return candidate.log_probability / candidate.new_tokens > previous.log_probability / previous.new_tokens

    def observe(self, evaluation: PrefixEvaluation) -> None:
        """Consume the pending prefix's full distribution, retaining its provenance.

        Invalid distributions leave the search state untouched. Character limits
        restrict expansion, not normalization, and STOP is never beam-pruned.
        """
        if self.pending_prefix is None:
            raise ValueError("no pending prefix to evaluate")
        probabilities = evaluation.probabilities
        if probabilities.keys() != self._vocabulary.keys():
            raise ValueError("evaluation must score exactly the fixed vocabulary")
        if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
            raise ValueError("probabilities must be finite and nonnegative")
        try:
            total = math.fsum(probabilities.values())
        except OverflowError:
            raise ValueError("probabilities must have a finite positive sum") from None
        if not math.isfinite(total) or total <= 0:
            raise ValueError("probabilities must have a finite positive sum")
        log_total = math.log(total)
        parent = self._frontier[self._index]
        has_live_child = False
        for label, token in self._vocabulary.items():
            probability = probabilities[label]
            if probability == 0:
                continue
            if token is not None and len(parent.prefix) + len(token) > self._max_length:
                continue
            prefix = parent.prefix if token is None else parent.prefix + token
            # Subtract logs instead of dividing first: tiny positive edges must
            # survive even when their normalized probability would underflow.
            log_probability = parent.log_probability + (math.log(probability) - log_total)
            if token is None:
                if self._completed is not None:
                    score = log_probability / (parent.new_tokens + 1)
                    if score <= self._completed.log_probability / self._completed.new_tokens:
                        continue
            else:
                has_live_child = True
                previous = self._children.get(prefix)
                if previous is not None and log_probability <= previous.log_probability:
                    continue
            child = _Node(
                prefix, log_probability, parent.new_tokens + 1, parent,
                BeamStep(parent.prefix, prefix, label, evaluation),
            )
            if token is None:
                self._completed = child
            else:
                # Reinsert a replacement so equal-score ties follow the winning
                # paths' actual frontier/vocabulary visitation order.
                if previous is not None:
                    del self._children[prefix]
                self._children[prefix] = child

        # A capped or dead-end fragment remains a valid partial result after its
        # evaluation. Expanded ancestors, unlike these leaves, are not candidates.
        if not has_live_child and self._better_mean(parent, self._terminal_partial):
            self._terminal_partial = parent
        self._index += 1
        if self._index == len(self._frontier):
            ordered = sorted(self._children.values(), key=lambda node: node.log_probability, reverse=True)
            if self._diversity == "token" and len(ordered) > self._beam_width:
                # Preserve the best continuation ending in each distinct action
                # before filling spare slots by probability. No token is masked,
                # and sequence scores remain the original model scores.
                represented: set[str] = set()
                retained: set[str] = set()
                for node in ordered:
                    assert node.step is not None
                    if node.step.label not in represented:
                        represented.add(node.step.label)
                        retained.add(node.prefix)
                        if len(retained) == self._beam_width:
                            break
                for node in ordered:
                    if len(retained) == self._beam_width:
                        break
                    retained.add(node.prefix)
                self._frontier = [node for node in ordered if node.prefix in retained]
            else:
                self._frontier = ordered[:self._beam_width] if len(ordered) > self._beam_width else ordered
            self._children = {}
            self._index = 0
            self._depth += 1

    def result(self) -> BeamResult:
        """Return a completion if available, otherwise the best available leaf."""
        selected = self._completed
        if selected is None:
            selected = self._terminal_partial
            for candidate in chain(islice(self._frontier, self._index, None), self._children.values()):
                if self._better_mean(candidate, selected):
                    selected = candidate
        if selected is None:
            selected = self._root
        steps: list[BeamStep] = []
        node = selected
        while node.step is not None:
            steps.append(node.step)
            assert node.parent is not None
            node = node.parent
        steps.reverse()
        return BeamResult(
            text=selected.prefix,
            completed=self._completed is not None,
            steps=tuple(steps),
            log_probability=selected.log_probability,
            score=selected.score,
        )

    def snapshot(self) -> dict:
        """Report pending/current live branches and the best completed branch."""
        return {
            "depth": self._depth,
            "active": [node.snapshot() for node in chain(
                islice(self._frontier, self._index, None), self._children.values(),
            )],
            "completed": None if self._completed is None else self._completed.snapshot(),
        }
