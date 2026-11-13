"""Unit tests for yaml.bounded -- the bounded safe config-import service.

All fixtures are plain strings built inline; no files are read.
Run with:  PYTHONPATH=lib python3 -m unittest tests/test_bounded_loader.py
"""

import unittest

import yaml
from yaml.bounded import (
    BoundedSafeLoader,
    ConfigImportError,
    ResourceLimits,
    load_all_config,
    load_config,
)


def alias_bomb(levels=8, fanout=10):
    """Classic billion-laughs: each level aliases the previous one `fanout`
    times.  The text is tiny; the logical expansion is fanout**levels."""
    lines = ['l0: &a0 [%s]' % ','.join('"x"' for _ in range(fanout))]
    for i in range(1, levels):
        refs = ','.join('*a%d' % (i - 1) for _ in range(fanout))
        lines.append('l%d: &a%d [%s]' % (i, i, refs))
    return '\n'.join(lines) + '\n'


VALID_CONFIG = """\
service:
  name: billing
  replicas: 3
defaults: &defaults
  retries: 2
  timeout: 30
primary:
  <<: *defaults
  timeout: 10
"""


class ValidInputTests(unittest.TestCase):

    def test_valid_config_roundtrip(self):
        result = load_config(VALID_CONFIG)
        self.assertEqual(result['service']['name'], 'billing')
        self.assertEqual(result['service']['replicas'], 3)
        # merge key: inherited retries, explicit timeout wins
        self.assertEqual(result['primary'], {'retries': 2, 'timeout': 10})

    def test_empty_and_scalar_documents(self):
        self.assertIsNone(load_config(''))
        self.assertEqual(load_config('just a string\n'), 'just a string')
        self.assertEqual(load_config('42\n'), 42)

    def test_moderate_alias_use_within_budget(self):
        text = 'a: &x [1, 2, 3]\nb: *x\nc: *x\n'
        result = load_config(text)
        self.assertEqual(result['a'], [1, 2, 3])
        self.assertIs(result['b'], result['c'])

    def test_load_all_documents(self):
        text = '---\na: 1\n---\nb: 2\n'
        self.assertEqual(load_all_config(text), [{'a': 1}, {'b': 2}])

    def test_single_load_rejects_multi_document_stream(self):
        with self.assertRaises(ConfigImportError):
            load_config('---\na: 1\n---\nb: 2\n')

    def test_document_count_limit(self):
        text = '---\na: 1\n---\nb: 2\n---\nc: 3\n'
        limits = ResourceLimits(max_documents=2)
        with self.assertRaises(ConfigImportError) as cm:
            load_all_config(text, limits)
        self.assertIn('documents', str(cm.exception))


class AliasAndDepthLimitTests(unittest.TestCase):

    def test_alias_bomb_rejected(self):
        with self.assertRaises(ConfigImportError) as cm:
            load_config(alias_bomb())
        self.assertIn('alias expansion', str(cm.exception))

    def test_alias_bomb_rejected_with_small_budget(self):
        limits = ResourceLimits(max_logical_nodes=500)
        with self.assertRaises(ConfigImportError):
            load_config(alias_bomb(levels=5, fanout=6), limits)

    def test_alias_bomb_text_is_tiny(self):
        # the guard must fire on expansion, not on input size
        self.assertLess(len(alias_bomb().encode('utf-8')), 4096)

    def test_deep_nesting_rejected_by_default_limit(self):
        text = '[' * 300 + ']' * 300 + '\n'
        with self.assertRaises(ConfigImportError) as cm:
            load_config(text)
        self.assertIn('depth', str(cm.exception))

    def test_deep_nesting_rejected_by_custom_limit(self):
        text = 'a:\n  b:\n    c:\n      d: 1\n'
        limits = ResourceLimits(max_depth=3)
        with self.assertRaises(ConfigImportError):
            load_config(text, limits)

    def test_deep_nesting_fails_cleanly_not_with_recursion_error(self):
        text = '[' * 100000 + ']' * 100000
        with self.assertRaises(ConfigImportError):
            load_config(text)


class ScalarAndInputLimitTests(unittest.TestCase):

    def test_oversized_scalar_rejected(self):
        text = 'key: ' + 'x' * 5000 + '\n'
        limits = ResourceLimits(max_scalar_bytes=1024)
        with self.assertRaises(ConfigImportError) as cm:
            load_config(text, limits)
        self.assertIn('scalar', str(cm.exception))

    def test_oversized_input_rejected_before_parsing(self):
        text = '# padding\n' * 100 + 'key: value\n'
        limits = ResourceLimits(max_input_bytes=100)
        with self.assertRaises(ConfigImportError) as cm:
            load_config(text, limits)
        self.assertEqual(cm.exception.kind, 'ResourceLimit')

    def test_unicode_scalars_roundtrip(self):
        text = (
            'name: "配置文件 🚀"\n'
            'emoji: "👍🏽"\n'
            'arabic: "مرحبا"\n'
            'combining: "café"\n'
            'multiline: |\n'
            '  多行\n'
            '  文本\n'
        )
        result = load_config(text)
        self.assertEqual(result['name'], '配置文件 🚀')
        self.assertEqual(result['emoji'], '👍🏽')
        self.assertEqual(result['arabic'], 'مرحبا')
        self.assertEqual(result['combining'], 'café')
        self.assertEqual(result['multiline'], '多行\n文本\n')

    def test_unicode_within_scalar_budget_counts_bytes_not_chars(self):
        # "界" is 3 bytes in UTF-8; 10 of them = 30 bytes
        text = 'key: "' + '界' * 10 + '"\n'
        self.assertEqual(load_config(text, ResourceLimits(max_scalar_bytes=30)),
                         {'key': '界' * 10})
        with self.assertRaises(ConfigImportError):
            load_config(text, ResourceLimits(max_scalar_bytes=29))


class DuplicateKeyTests(unittest.TestCase):

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ConfigImportError) as cm:
            load_config('a: 1\nb: 2\na: 3\n')
        self.assertIn('duplicate key', str(cm.exception))
        self.assertEqual(cm.exception.line, 3)

    def test_duplicate_keys_in_nested_mapping_rejected(self):
        text = 'outer:\n  inner: 1\n  inner: 2\n'
        with self.assertRaises(ConfigImportError):
            load_config(text)

    def test_duplicate_keys_allowed_by_explicit_opt_in(self):
        limits = ResourceLimits(allow_duplicate_keys=True)
        result = load_config('a: 1\na: 2\n', limits)
        self.assertEqual(result, {'a': 2})

    def test_merge_key_override_is_not_a_duplicate(self):
        text = (
            'base: &b\n'
            '  x: 1\n'
            '  y: 2\n'
            'derived:\n'
            '  <<: *b\n'
            '  y: 3\n'
        )
        result = load_config(text)
        self.assertEqual(result['derived'], {'x': 1, 'y': 3})


class RecursionAndTagTests(unittest.TestCase):

    def test_recursive_sequence_rejected(self):
        with self.assertRaises(ConfigImportError) as cm:
            load_config('a: &x\n  - *x\n')
        self.assertIn('recursive', str(cm.exception))

    def test_recursive_mapping_rejected(self):
        with self.assertRaises(ConfigImportError):
            load_config('root: &r\n  self: *r\n')

    def test_unknown_tag_rejected(self):
        with self.assertRaises(ConfigImportError) as cm:
            load_config('service: !internal/secret {token: abc}\n')
        self.assertEqual(cm.exception.kind, 'ConstructorError')
        self.assertEqual(cm.exception.line, 1)

    def test_python_object_tag_never_constructed(self):
        text = '!!python/object/new:os.system\nargs: ["echo pwned"]\n'
        with self.assertRaises(ConfigImportError):
            load_config(text)

    def test_python_module_tag_rejected_in_mapping(self):
        text = 'callback: !!python/name:os.system ""\n'
        with self.assertRaises(ConfigImportError):
            load_config(text)


class ErrorReportingTests(unittest.TestCase):

    def test_error_preserves_line_and_column(self):
        # a tab character where indentation is expected: the scanner
        # reports it at line 3, column 1
        text = 'good: 1\nalso_good: 2\n\tbad: 3\n'
        with self.assertRaises(ConfigImportError) as cm:
            load_config(text)
        exc = cm.exception
        self.assertEqual(exc.line, 3)
        self.assertEqual(exc.column, 1)
        self.assertIn('line 3', str(exc))

    def test_error_does_not_echo_sensitive_content(self):
        text = (
            'database:\n'
            '  password: "hunter2-super-secret"\n'
            '  host: db.internal\n'
            'broken: !unknown [1, 2\n'
        )
        with self.assertRaises(ConfigImportError) as cm:
            load_config(text)
        for rendered in (str(cm.exception), repr(cm.exception)):
            self.assertNotIn('hunter2', rendered)
            self.assertNotIn('db.internal', rendered)

    def test_scan_error_does_not_echo_sensitive_content(self):
        text = 'password: "hunter2-super-secret"\nbad:\n\t- tabbed\n'
        with self.assertRaises(ConfigImportError) as cm:
            load_config(text)
        self.assertNotIn('hunter2', str(cm.exception))
        self.assertEqual(cm.exception.line, 3)

    def test_error_message_is_single_line_and_bounded(self):
        with self.assertRaises(ConfigImportError) as cm:
            load_config('a: 1\na: 2\n')
        message = str(cm.exception)
        self.assertNotIn('\n', message)
        self.assertLessEqual(len(message), 300)


class StateAndIsolationTests(unittest.TestCase):

    def test_repeated_failures_do_not_accumulate_state(self):
        bomb = alias_bomb()
        messages = set()
        for _ in range(3):
            with self.assertRaises(ConfigImportError) as cm:
                load_config(bomb)
            messages.add(str(cm.exception))
        # identical, deterministic failure each time
        self.assertEqual(len(messages), 1)

    def test_load_succeeds_after_failure(self):
        with self.assertRaises(ConfigImportError):
            load_config('a: 1\na: 2\n')
        self.assertEqual(load_config('ok: true\n'), {'ok': True})
        with self.assertRaises(ConfigImportError):
            load_config('[' * 500 + ']' * 500)
        self.assertEqual(load_config(VALID_CONFIG)['service']['replicas'], 3)

    def test_limits_are_immutable(self):
        limits = ResourceLimits(max_depth=5)
        with self.assertRaises(Exception):
            limits.max_depth = 1000

    def test_stock_safeloader_not_tightened_by_our_policy(self):
        # run bounded loads first so any leaked state would show up
        with self.assertRaises(ConfigImportError):
            load_config('a: 1\na: 2\n')
        # stock safe_load keeps its own (lenient) duplicate-key behavior
        self.assertEqual(yaml.safe_load('a: 1\na: 2\n'), {'a': 2})
        # stock safe_load still allows recursive structures
        result = yaml.safe_load('root: &r\n  self: *r\n')
        self.assertIs(result['root']['self'], result['root'])

    def test_our_policy_does_not_relax_safeloader(self):
        # unknown tags still fail on the stock SafeLoader with its own
        # error type, not ours
        with self.assertRaises(yaml.YAMLError) as cm:
            yaml.safe_load('x: !nope 1\n')
        self.assertNotIsInstance(cm.exception, ConfigImportError)

    def test_no_constructors_registered_on_shared_classes(self):
        before = dict(yaml.SafeLoader.yaml_constructors)
        load_config(VALID_CONFIG)
        with self.assertRaises(ConfigImportError):
            load_config(alias_bomb())
        self.assertEqual(dict(yaml.SafeLoader.yaml_constructors), before)
        self.assertFalse(
            BoundedSafeLoader.__dict__.get('yaml_constructors'))


if __name__ == '__main__':
    unittest.main()
