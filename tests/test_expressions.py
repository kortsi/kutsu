# noqa
# type:ignore
import pytest

from kutsu.expressions import (
    Add,
    And,
    Del,
    DelIfNone,
    Div,
    Eq,
    FloorDiv,
    GetAttr,
    GetItem,
    Gt,
    Gte,
    If,
    In,
    IsFalse,
    IsFalsy,
    IsNone,
    IsNotFalse,
    IsNotFalsy,
    IsNotNone,
    IsNotTrue,
    IsNotTruthy,
    IsTrue,
    IsTruthy,
    Json,
    Lt,
    Lte,
    Mod,
    Mul,
    Ne,
    Not,
    NotIn,
    Or,
    Pow,
    Raise,
    Secret,
    Select,
    Sub,
    Var,
    VarAnd,
    VarOr,
    subst_data,
    subst_str,
)
from kutsu.state import State

from .generate_tests import pytest_generate_tests


class TestSubstData:

    def test_subst_data(self, arg0, state, output):
        assert subst_data(arg0, state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': [
                'True', '1', '"yes"', '["true"]', 'False', '0', '""', '[]', 'None',
                '{"a": 1, "b": "2"}'
            ],
            'arg0': [True, 1, 'yes', ['true'], False, 0, '', [], None,
                     dict(a=1, b='2')],
            'state': State(),
        },
        'Identity': {
            'output': [True, 1, 'yes', ['true'], False, 0, '', [], None,
                       dict(a=1, b='2')],
        },
    }


class TestState:

    def test_subst_data(self, arg0, state, output):
        assert subst_data(arg0, state) == output

    test_scenarios = {
        'DEFAULTS': {
            'arg0': {
                'a': 1,
                'b': '2',
                'c': Var('x'),
                'd': '${x}'
            },
        },
        'WithState': {
            'state': State(x=1),
            'output': {
                'a': 1,
                'b': "2",
                'c': 1,
                'd': '1'
            },
        },
        'NoState': {
            'state': State(),
            'output': {
                'a': 1,
                'b': "2",
                'c': None,
                'd': ''
            },
        },
    }


class TestUnaryBool:

    def test_subst_data(self, arg0, function, state, output):
        assert subst_data(function(arg0), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': ['True', '1', '"yes"', '["true"]', 'False', '0', '""', '[]', 'None'],
            'arg0': [True, 1, 'yes', ['true'], False, 0, '', [], None],
            'state': State(),
        },
        'IsTrue': {
            'function': IsTrue,
            'output': [True, False, False, False, False, False, False, False, False],
        },
        'IsTruthy': {
            'function': IsTruthy,
            'output': [True, True, True, True, False, False, False, False, False]
        },
        'IsFalse': {
            'function': IsFalse,
            'output': [False, False, False, False, True, False, False, False, False]
        },
        'IsFalsy': {
            'function': IsFalsy,
            'output': [False, False, False, False, True, True, True, True, True]
        },
        'IsNotTrue': {
            'function': IsNotTrue,
            'output': [False, True, True, True, True, True, True, True, True]
        },
        'IsNotTruthy': {
            'function': IsNotTruthy,
            'output': [False, False, False, False, True, True, True, True, True]
        },
        'IsNotFalse': {
            'function': IsNotFalse,
            'output': [True, True, True, True, False, True, True, True, True]
        },
        'IsNotFalsy': {
            'function': IsNotFalsy,
            'output': [True, True, True, True, False, False, False, False, False]
        },
        'Not': {
            'function': Not,
            'output': [False, False, False, False, True, True, True, True, True]
        },
        'IsNone': {
            'function': IsNone,
            'output': [False, False, False, False, False, False, False, False, True]
        },
        'IsNotNone': {
            'function': IsNotNone,
            'output': [True, True, True, True, True, True, True, True, False]
        },
    }


class TestEqual:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': ['1', '"1"', '["a", "b"]', '{"x": 1, "y": -1}'],
            'state': State(),
            'arg0': [1, '1', ['a', 'b'], dict(x=1, y=-1)],
            'arg1': [1, '1', ['a', 'b'], dict(x=1, y=-1)],
        },
        'EqTrue': {
            'function': Eq,
            'output': [True] * 4
        },
        'NeFalse': {
            'function': Ne,
            'output': [False] * 4
        },
    }


class TestNotEqual:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': ['1-0', '"1"-1', '["a", "b"]-("a", "b")', '{"x": 1, "y": -1}-None'],
            'state': State(),
            'arg0': [1, '1', ['a', 'b'], dict(x=1, y=-1)],
            'arg1': [0, 1, ('a', 'b'), None],
        },
        'EqFalse': {
            'function': Eq,
            'output': [False] * 4
        },
        'NeTrue': {
            'function': Ne,
            'output': [True] * 4
        },
    }


class TestEqualVar:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': ['True', '1', '"yes"', '["true"]', 'False', '0', '""', '[]', 'None'],
            'state': State(A=True, B=1, C='yes', D=['true'], a=False, b=0, c='', d=[], e=None),
            'arg0': [Var(x) for x in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd', 'e']],
            'arg1': [True, 1, 'yes', ['true'], False, 0, '', [], None],
        },
        'EqTrue': {
            'function': Eq,
            'output': [True] * 9
        },
        'NeFalse': {
            'function': Ne,
            'output': [False] * 9
        },
    }


class TestNotEqualVar:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': ['True', '1', '"yes"', '["true"]', 'False', '0', '""', '[]', 'None'],
            'state': State(A=True, B=1, C='yes', D=['true'], a=False, b=0, c='', d=[], e=None),
            'arg0': [Var(x) for x in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd', 'e']],
            'arg1': [False, 2, 'no', ['false'], True, 1, 'x', ['a'], True],
        },
        'EqFalse': {
            'function': Eq,
            'output': [False] * 9
        },
        'NeTrue': {
            'function': Ne,
            'output': [True] * 9
        },
    }


class TestComparison:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': [
                'Var("B")-0', 'Var("B")-1', 'Var("B")-2"', 'Var("B")-Var("b")', 'Var("B")-Var("C")'
            ],
            'state': State(B=1, C='yes', b=0, c=''),
            'arg0': [Var('B'), Var('B'), Var('B'), Var('B'),
                     Var('C')],
            'arg1': [0, 1, 2, Var('b'), Var('c')],
        },
        'Gt': {
            'function': Gt,
            'output': [True, False, False, True, True],
        },
        'Gte': {
            'function': Gte,
            'output': [True, True, False, True, True],
        },
        'Lt': {
            'function': Lt,
            'output': [False, False, True, False, False],
        },
        'Lte': {
            'function': Lte,
            'output': [False, True, True, False, False],
        },
    }


class TestComparisonOperator:

    def test_type(self, arg0, type_, state, output):
        assert isinstance(arg0, type_)

    def test_subst_data(self, arg0, type_, state, output):
        assert subst_data(arg0, state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(a='a', b='b', zero=0, one=1, two=2, three=3),
        },
        'Eq': {
            'type_': Eq,
            'arg0': [
                Var('a') == Var('a'),
                Var('a') == Var('b'),
                Var('a') == 'a',
                Var('a') == 'b',
            ],
            'output': [True, False, True, False],
        },
        'Ne': {
            'type_': Ne,
            'arg0': [
                Var('a') != Var('a'),
                Var('a') != Var('b'),
                Var('a') != 'a',
                Var('a') != 'b',
            ],
            'output': [False, True, False, True],
        },
        'Gt': {
            'type_': Gt,
            'arg0': [
                Var('zero') > -1,
                Var('zero') > 0,
                Var('zero') > 1,
                Var('two') > Var('one'),
                Var('two') > Var('two'),
                Var('two') > Var('three')
            ],
            'output': [True, False, False, True, False, False],
        },
        'Gte': {
            'type_': Gte,
            'arg0': [
                Var('zero') >= -1,
                Var('zero') >= 0,
                Var('zero') >= 1,
                Var('two') >= Var('one'),
                Var('two') >= Var('two'),
                Var('two') >= Var('three')
            ],
            'output': [True, True, False, True, True, False],
        },
        'Lt': {
            'type_': Lt,
            'arg0': [
                Var('zero') < -1,
                Var('zero') < 0,
                Var('zero') < 1,
                Var('two') < Var('one'),
                Var('two') < Var('two'),
                Var('two') < Var('three')
            ],
            'output': [False, False, True, False, False, True],
        },
        'Lte': {
            'type_': Lte,
            'arg0': [
                Var('zero') <= -1,
                Var('zero') <= 0,
                Var('zero') <= 1,
                Var('two') <= Var('one'),
                Var('two') <= Var('two'),
                Var('two') <= Var('three')
            ],
            'output': [False, True, True, False, True, True],
        },
    }


class TestContaining:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'IDS': [
                '1-[1,2,3]', '1-[0,2,3]', '[1]-[1,2,3]"', 'Var("a")-Var("b")', 'Var("c")-Var("b")'
            ],
            'state': State(a='a', b=[Var('a'), 'b', 'c']),
            'arg0': [1, 1, [1], Var('a'), Var('c')],
            'arg1': [[1, 2, 3], [0, 2, 3], [1, 2, 3],
                     Var('b'), Var('b')],
        },
        'In': {
            'function': In,
            'output': [True, False, False, True, False],
        },
        'NotIn': {
            'function': NotIn,
            'output': [False, True, True, False, True],
        },
    }


class TestAlgebra:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(A=True, B=1, C='yes', D=['true'], a=False, b=0, c='', d=[], e=None),
        },
        'Add': {
            'function': Add,
            'IDS': ['1+1', '"a"+"b"', '[1]+[2]', 'Var("D")+["false"]'],
            'arg0': [1, 'a', [1], Var('D')],
            'arg1': [1, 'b', [2], ['false']],
            'output': [2, 'ab', [1, 2], ['true', 'false']],
        },
        'Mul': {
            'function': Mul,
            'IDS': ['2*2', '"a"*2', "[2]*2"],
            'arg0': [2, "a", [2]],
            'arg1': [2, 2, 2],
            'output': [4, "aa", [2, 2]],
        },
        'Sub-1-1': {
            'function': Sub,
            'arg0': 1,
            'arg1': 1,
            'output': 0,
        },
        'Div-2-2': {
            'function': Div,
            'arg0': 2,
            'arg1': 2,
            'output': 1,
        },
        'FloorDiv-5-2': {
            'function': FloorDiv,
            'arg0': 5,
            'arg1': 2,
            'output': 2,
        },
        'Mod-5-2': {
            'function': Mod,
            'arg0': 5,
            'arg1': 2,
            'output': 1,
        },
        'Pow-2-10': {
            'function': Pow,
            'arg0': 2,
            'arg1': 10,
            'output': 1024,
        },
    }


class TestAlgebraOperator:

    def test_type(self, arg0, type_, state, output):
        assert isinstance(arg0, type_)

    def test_subst_data(self, arg0, type_, state, output):
        assert subst_data(arg0, state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(a='a', b='b', zero=0, one=1, two=2, three=3),
        },
        'Add': {
            'type_': Add,
            'arg0': [
                Var('a') + Var('b'),
                Var('one') + Var('one'),
                Var('one') + 1, 1 + Var('one'),
                Var('one') + Var('two') * Var('zero')
            ],
            'output': ['ab', 2, 2, 2, 1],
        },
        'Sub': {
            'type_': Sub,
            'arg0': [
                Var('one') - Var('one'),
                Var('one') - 1, 1 - Var('one'),
                Var('one') - Var('two') * Var('zero')
            ],
            'output': [0, 0, 0, 1],
        },
        'Mul': {
            'type_': Mul,
            'arg0': [
                Var('a') * Var('two'),
                Var('one') * Var('two'),
                Var('one') * 2, 2 * Var('one'), (Var('one') + Var('two')) * Var('two')
            ],
            'output': ['aa', 2, 2, 2, 6],
        },
        'Div': {
            'type_': Div,
            'arg0': [
                Var('two') / Var('two'),
                Var('two') / 2, 2 / Var('two'), (Var('one') + Var('one')) / Var('two')
            ],
            'output': [1, 1, 1, 1],
        },
        'Mod': {
            'type_': Mod,
            'arg0': [
                Var('three') % Var('two'),
                Var('three') % 2, 3 % Var('two'), (Var('one') + Var('two')) % Var('two')
            ],
            'output': [1, 1, 1, 1],
        },
        'FloorDiv': {
            'type_': FloorDiv,
            'arg0': [
                Var('three') // Var('two'),
                Var('three') // 2, 3 // Var('two'), (Var('one') + Var('two')) // Var('two')
            ],
            'output': [1, 1, 1, 1],
        },
        'Pow': {
            'type_': Pow,
            'arg0': [
                Var('two')**Var('two'),
                Var('two')**2, 2**Var('two'), (Var('one') + Var('one'))**Var('two')
            ],
            'output': [4, 4, 4, 4],
        },
    }


class TestVarSubst:

    def test_subst_data(self, arg0, function, state, output):
        assert subst_data(function(arg0), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(
                a0='find me', a1=Var('a0'), a2=Var('a1'), a3='${a2}', a4='${a3}', a5=Var('a4')
            ),
            'function': Var,
        },
        'Var': {
            'IDS': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'missing'],
            'arg0': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'missing'],
            'output': ['find me'] * 6 + [None],
        },
    }


class TestStrSubst:

    def test_subst_data(self, arg0, state, output):
        assert subst_data(arg0, state) == output

    def test_subst_str(self, arg0, state, output):
        assert subst_str(arg0, state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(
                a0='find me', a1=Var('a0'), a2=Var('a1'), a3='${a2}', a4='${a3}', a5=Var('a4')
            ),
        },
        'Var': {
            'IDS': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'missing'],
            'arg0': ['${a0}', '${a1}', '${a2}', '${a3}', '${a4}', '${a5}', '${missing}'],
            'output': ['find me'] * 6 + [''],
        },
    }


def test_getitem():
    state = State(l=[1, 2, 3], ll=[[4, 5], [6]])

    assert subst_data(GetItem(Var('l'), 0), state) == 1
    assert subst_data(GetItem(Var('l'), 1), state) == 2
    assert subst_data(GetItem(Var('l'), 2), state) == 3

    assert subst_data(Var('l')[0], state) == 1
    assert subst_data(Var('l')[1], state) == 2
    assert subst_data(Var('l')[2], state) == 3

    assert subst_data(Var('ll')[0][0], state) == 4
    assert subst_data(Var('ll')[0][1], state) == 5
    assert subst_data(Var('ll')[1][0], state) == 6


# def test_getattr():
#     state = State(s=State(a='A', b='B'), ss=State(sss=State(c='C')))
#
#     assert subst_data(GetAttr(Var('s'), 'a'), state) == 'A'
#     assert subst_data(GetAttr(Var('s'), 'b'), state) == 'B'
#
#     assert subst_data(Var('s').a, state) == 'A'
#     assert subst_data(Var('s').b, state) == 'B'
#
#     assert subst_data(Var('ss').sss.c, state) == 'C'


def test_secret():
    state = State(secret=Secret('secret_value'))

    assert subst_data(Var('secret'), state, mask_secrets=False) == 'secret_value'
    assert 'secret_value' not in subst_data(Var('secret'), state, mask_secrets=True)
    assert 'secret_value' in subst_data('a ${secret}', state, mask_secrets=False)
    assert 'secret_value' not in subst_data('a ${secret}', state, mask_secrets=True)
    assert 'secret_value' in subst_str('a ${secret}', state, mask_secrets=False)
    assert 'secret_value' not in subst_str('a ${secret}', state, mask_secrets=True)


def test_secret_add():
    state = State(secret=Secret('secret_value'))

    assert 'secret_value' not in subst_data(Add(Var('secret'), ''), state, mask_secrets=True)


def test_secret_mul():
    state = State(secret=Secret('secret_value'))

    assert 'secret_value' not in subst_data(Mul(Var('secret'), 2), state, mask_secrets=True)


class TestLogical:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(a=True, b=False, c='c', d=''),
            'IDS': ['Var("a")', 'Var("b")', 'Var("c")', 'Var("d")', 'Var("missing")'],
            'arg0': [Var('a'), Var('b'), Var('c'),
                     Var('d'), Var('missing')],
            'arg1': [1, 1, 1, 1, 1],
        },
        'And': {
            'function': And,
            'output': [1, None, 1, None, None]
        },
        'Or': {
            'function': Or,
            'output': [True, 1, 'c', 1, 1]
        },
    }


class TestLogicalOperator:

    def test_type(self, arg0, type_, state, output):
        assert isinstance(arg0, type_)

    def test_subst_data(self, arg0, type_, state, output):
        assert subst_data(arg0, state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(a=True, b=False, c='c', d=''),
        },
        'And': {
            'type_': And,
            'arg0': [
                Var('a') & Var('c'),
                Var('b') & Var('c'),
                Var('a') & 'yes',
                Var('b') & 'no',
                Var('c') & 'yes',
                Var('d') & 'no',
                True & Var('c'),
                False & Var('c'),
            ],
            'output': ['c', None, 'yes', None, 'yes', None, 'c', None],
        },
        'Or': {
            'type_': Or,
            'arg0': [
                Var('a') | Var('c'),
                Var('b') | Var('c'),
                Var('a') | 'yes',
                Var('b') | 'no',
                Var('c') | 'yes',
                Var('d') | 'no',
                True | Var('c'),
                False | Var('c'),
            ],
            'output': [True, 'c', True, 'no', 'c', 'no', True, 'c'],
        },
    }


class TestLogicalVar:

    def test_subst_data(self, arg0, arg1, function, state, output):
        assert subst_data(function(arg0, arg1), state) == output

    test_scenarios = {
        'DEFAULTS': {
            'state': State(a=True, b=False, c='c', d=''),
            'IDS': ['a', 'b', 'c', 'd', 'missing'],
            'arg0': ['a', 'b', 'c', 'd', 'missing'],
            'arg1': [1, 1, 1, 1, 1],
        },
        'VarAnd': {
            'function': VarAnd,
            'output': [1, None, 1, None, None]
        },
        'VarOr': {
            'function': VarOr,
            'output': [True, 1, 'c', 1, 1]
        },
    }


def test_del_if_none():
    state = State(this_is_none=None, not_none='not None')

    assert isinstance(subst_data(DelIfNone(Var('this_is_none')), state), Del)
    assert subst_data(DelIfNone(Var('not_none')), state) == 'not None'


def test_del_if_none_operator():
    node = ~Var('x')
    assert isinstance(node, DelIfNone)
    assert isinstance(subst_data(node, State()), Del)
    assert subst_data(node, State(x=4)) == 4


def test_del():
    assert subst_data({'a': Del()}, State()) == {}
    assert subst_data({'a': Del(), 'b': 'keep'}, State()) == {'b': 'keep'}
    assert subst_data([Del()], State()) == []
    assert subst_data(['a', Del(), 'b'], State()) == ['a', 'b']


def test_json():
    assert subst_data(Json({'a': (1, 2, 3)}), State()) == {'a': [1, 2, 3]}
    assert subst_data(Json({'a': {1, 2, 3}}), State()) == {'a': [1, 2, 3]}


def test_raise():
    with pytest.raises(RuntimeError):
        subst_data(Raise('Error'), State())


def test_if():
    state = State(A=True, B=1, C='yes', a=False, b=0)
    assert subst_data(If('A', 1, 2), state) == 1
    assert subst_data(If('a', 1, 2), state) == 2
    assert subst_data(If(Var('A'), 1, 2), state) == 1
    assert subst_data(If(Var('a'), 1, 2), state) == 2
    assert subst_data(If(Gt(Var('B'), 0), 1, 2), state) == 1
    assert subst_data(If(Gt(Var('b'), 0), 1, 2), state) == 2
    assert subst_data(If('A', Var('C'), Var('b')), state) == 'yes'
    assert subst_data(If('a', Var('C'), Var('b')), state) == 0


def test_select():
    state = State(B=1, C='yes', b=0, e=None)
    selections = {1: 'one', 'yes': 'positive', 0: 'zero'}
    assert subst_data(Select('B', selections, 'default'), state) == 'one'
    assert subst_data(Select('C', selections, 'default'), state) == 'positive'
    assert subst_data(Select('b', selections, 'default'), state) == 'zero'
    assert subst_data(Select('e', selections, 'default'), state) == 'default'
    assert subst_data(Select('missing', selections, 'default'), state) == 'default'
    # selections[True] will for some reason find key 1
    # selections[False] will for some reason find key 0