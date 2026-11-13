"""
Bounded, safe YAML loading for untrusted configuration input.

This module implements a config-import service with explicit resource and
type boundaries.  All policies are enforced per load, on a dedicated loader
instance, so repeated loads of the same input never accumulate global state
and the stock ``SafeLoader``/``Loader`` classes are neither tightened nor
relaxed.

Policy summary (see ``ResourceLimits``):

* alias expansion: the *logical* node count (each alias reference counts the
  full size of the anchored subtree) is capped, which defuses exponential
  "alias bomb" documents before they allocate memory;
* nesting depth: composition depth is capped, so deeply nested collections
  fail with a clean error instead of a ``RecursionError``;
* scalar size: each scalar is capped in bytes (UTF-8);
* input size: the raw input is capped in bytes before parsing starts;
* duplicate keys: duplicate explicit keys in a mapping are rejected
  (merge keys ``<<`` keep their standard override semantics);
* recursive references: an alias that refers to an anchor which is still
  being composed (i.e. a self-referential structure) is rejected;
* unknown tags: any tag without a safe constructor (including all
  ``!!python/...`` tags) is rejected -- no objects are ever constructed.

Errors are reported as ``ConfigImportError`` carrying the 1-based line and
column of the problem, but never the source snippet: the offending document
text is not echoed back, so secrets in the configuration do not leak into
logs or error responses.
"""

__all__ = [
    'ResourceLimits',
    'DEFAULT_LIMITS',
    'ConfigImportError',
    'BoundedSafeLoader',
    'load_config',
    'load_all_config',
]

import collections.abc
from dataclasses import dataclass

from .composer import ComposerError
from .constructor import BaseConstructor, ConstructorError
from .error import YAMLError
from .events import AliasEvent, ScalarEvent
from .loader import SafeLoader
from .nodes import MappingNode, ScalarNode, SequenceNode

#: Maximum length of the problem summary embedded in an error message.
#: Keeps messages single-line and bounded even if a parser problem string
#: quotes a character or name from the input.
_MAX_PROBLEM_LEN = 160


@dataclass(frozen=True)
class ResourceLimits:
    """Resource and type boundaries for one config import.

    The defaults are conservative; pass a custom instance to ``load_config``
    / ``load_all_config`` / ``BoundedSafeLoader`` to tune them.  Instances
    are immutable, so limits cannot be widened mid-load.
    """

    #: Maximum size of the raw input, in bytes (UTF-8).
    max_input_bytes: int = 8 * 1024 * 1024
    #: Maximum number of documents in a stream (``load_all_config``).
    max_documents: int = 16
    #: Maximum logical node count.  Every composed node counts 1; every
    #: alias reference counts the size of the anchored subtree, so
    #: exponential alias expansion trips this limit after composing only a
    #: handful of physical nodes.
    max_logical_nodes: int = 100_000
    #: Maximum composition nesting depth.
    max_depth: int = 100
    #: Maximum size of a single scalar value, in bytes (UTF-8).
    max_scalar_bytes: int = 1024 * 1024
    #: Whether a mapping may repeat an explicit key.  Default: reject.
    allow_duplicate_keys: bool = False
    #: Whether self-referential (recursive) structures are allowed.
    #: Default: reject.
    allow_recursive: bool = False


#: Shared immutable default limits.
DEFAULT_LIMITS = ResourceLimits()


class ConfigImportError(Exception):
    """Sanitized load failure.

    Carries ``kind`` (the originating error category), a short single-line
    ``problem`` summary, and 1-based ``line``/``column`` when the parser
    reported a position.  The source document is never included.
    """

    def __init__(self, problem, line=None, column=None, kind='ConfigError'):
        self.kind = kind
        self.problem = problem
        self.line = line
        self.column = column
        super().__init__(str(self))

    def __str__(self):
        problem = ' '.join(str(self.problem).split())
        if len(problem) > _MAX_PROBLEM_LEN:
            problem = problem[:_MAX_PROBLEM_LEN] + '...'
        message = '%s: %s' % (self.kind, problem)
        if self.line is not None:
            message += ' (line %d, column %d)' % (self.line, self.column)
        return message


def _convert_error(exc):
    """Convert a MarkedYAMLError to a ConfigImportError, keeping only the
    problem summary and position -- never the context snippet."""
    mark = getattr(exc, 'problem_mark', None)
    problem = getattr(exc, 'problem', None) or type(exc).__name__
    line = column = None
    if mark is not None:
        line = mark.line + 1
        column = mark.column + 1
    return ConfigImportError(problem, line=line, column=column,
                             kind=type(exc).__name__)


class BoundedSafeLoader(SafeLoader):
    """A SafeLoader that enforces ResourceLimits.

    All counters live on the instance; nothing is written to class-level or
    module-level state, and no constructors are registered on any shared
    loader class.
    """

    def __init__(self, stream, limits=None):
        self._limits = limits if limits is not None else DEFAULT_LIMITS
        self._depth = 0
        self._logical_nodes = 0
        self._subtree_sizes = {}
        self._open_anchors = set()
        super().__init__(stream)

    # -- compose-time guards (before any object is constructed) ----------

    def compose_node(self, parent, index):
        limits = self._limits
        if self.check_event(AliasEvent):
            event = self.peek_event()
            if not limits.allow_recursive and event.anchor in self._open_anchors:
                raise ComposerError(None, None,
                        "found recursive reference through anchor %r "
                        "(recursive structures are disabled)" % event.anchor,
                        event.start_mark)
            node = super().compose_node(parent, index)
            # An alias re-uses the anchored subtree: charge its full size
            # against the logical node budget.
            self._account_nodes(self._subtree_sizes.get(node, 1),
                                event.start_mark)
            return node

        event = self.peek_event()
        if isinstance(event, ScalarEvent) and event.value is not None:
            size = len(event.value.encode('utf-8'))
            if size > limits.max_scalar_bytes:
                raise ComposerError(None, None,
                        "scalar value of %d bytes exceeds the limit of %d bytes"
                        % (size, limits.max_scalar_bytes),
                        event.start_mark)
        if self._depth >= limits.max_depth:
            raise ComposerError(None, None,
                    "nesting depth exceeds the limit of %d levels"
                    % limits.max_depth,
                    event.start_mark)

        anchor = getattr(event, 'anchor', None)
        opened = False
        if anchor is not None and not isinstance(event, ScalarEvent):
            # The anchored collection is "open" while its children are
            # composed; an alias pointing at it here is recursive.
            self._open_anchors.add(anchor)
            opened = True
        self._depth += 1
        try:
            node = super().compose_node(parent, index)
        finally:
            self._depth -= 1
            if opened:
                self._open_anchors.discard(anchor)

        self._account_nodes(1, node.start_mark)
        if isinstance(node, ScalarNode):
            size = 1
        elif isinstance(node, SequenceNode):
            size = 1 + sum(self._subtree_sizes.get(child, 1)
                           for child in node.value)
        else:  # MappingNode
            size = 1 + sum(self._subtree_sizes.get(key, 1)
                           + self._subtree_sizes.get(value, 1)
                           for key, value in node.value)
        self._subtree_sizes[node] = size
        return node

    def _account_nodes(self, count, mark):
        self._logical_nodes += count
        if self._logical_nodes > self._limits.max_logical_nodes:
            raise ComposerError(None, None,
                    "document expands to more than %d nodes "
                    "(alias expansion limit)" % self._limits.max_logical_nodes,
                    mark)

    # -- construct-time guards --------------------------------------------

    def construct_object(self, node, deep=False):
        if not self._limits.allow_recursive and node in self.recursive_objects:
            raise ConstructorError(None, None,
                    "found recursive structure "
                    "(recursive structures are disabled)",
                    node.start_mark)
        return super().construct_object(node, deep)

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            return super().construct_mapping(node, deep)
        # Snapshot the explicit key nodes before flatten_mapping() splices
        # merged (<<) pairs in; duplicate detection applies to explicit
        # keys only, so merge-override semantics are preserved.
        explicit_ids = {id(key_node) for key_node, _ in node.value}
        self.flatten_mapping(node)
        if self._limits.allow_duplicate_keys:
            return BaseConstructor.construct_mapping(self, node, deep)
        mapping = {}
        seen_explicit = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, collections.abc.Hashable):
                raise ConstructorError("while constructing a mapping",
                        node.start_mark, "found unhashable key",
                        key_node.start_mark)
            if id(key_node) in explicit_ids:
                if key in seen_explicit:
                    raise ConstructorError("while constructing a mapping",
                            node.start_mark, "found duplicate key",
                            key_node.start_mark)
                seen_explicit.add(key)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _check_input_size(text, limits):
    if isinstance(text, bytes):
        size = len(text)
    else:
        size = len(text.encode('utf-8'))
    if size > limits.max_input_bytes:
        raise ConfigImportError(
                "input of %d bytes exceeds the limit of %d bytes"
                % (size, limits.max_input_bytes),
                kind='ResourceLimit')


def load_config(text, limits=None):
    """Load a single YAML document from a string, enforcing ``limits``.

    Returns plain safe types (dict/list/str/int/...).  Raises
    ``ConfigImportError`` on any violation or parse error.
    """
    limits = limits if limits is not None else DEFAULT_LIMITS
    _check_input_size(text, limits)
    loader = BoundedSafeLoader(text, limits)
    try:
        return loader.get_single_data()
    except YAMLError as exc:
        raise _convert_error(exc) from None
    except RecursionError:
        raise ConfigImportError("structure too deeply nested to construct",
                                kind='ResourceLimit') from None
    finally:
        loader.dispose()


def load_all_config(text, limits=None):
    """Load every document of a YAML stream, enforcing ``limits``
    (including ``max_documents``).  Returns a list of plain safe types."""
    limits = limits if limits is not None else DEFAULT_LIMITS
    _check_input_size(text, limits)
    loader = BoundedSafeLoader(text, limits)
    try:
        documents = []
        while loader.check_data():
            if len(documents) >= limits.max_documents:
                raise ConfigImportError(
                        "stream contains more than %d documents"
                        % limits.max_documents,
                        kind='ResourceLimit')
            documents.append(loader.get_data())
        return documents
    except YAMLError as exc:
        raise _convert_error(exc) from None
    except RecursionError:
        raise ConfigImportError("structure too deeply nested to construct",
                                kind='ResourceLimit') from None
    finally:
        loader.dispose()
