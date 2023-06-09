from typing import Any
import re
import sys

single_len_symbols = ["-", "+", "*", "/", "%", "^", "(", ")", "<", ">"]
boolean_symbols = ["&&", "||", "!"]
double_len_symbols = ["++", "--", "==", "!=", "<=", ">="]
op_equals_symbols = ["+=", "-=", "*=", "/=", "%=", "^=", "!="]
bool_equals_symbols = ["&&=", "||="]
assign_symbols = ["="]
keywords = ["print"]

disj_symbols = ["+", "-"]
conj_symbols = ["*", "/", "%"]
power_symbols = ["^"]
neg_symbols = ["-"]
incr_or_decr_symbols = ["--", "++"]
relational_symbols = ["<", ">", "<=", ">=", "==", "!="]


class token():
    typ: str
    val: str

    def __init__(self, typ, val):
        """
        >>> token('sym', '(')
        token('sym', '(')
        """
        self.typ = typ
        self.val = val

    def __repr__(self):
        return f'token({self.typ!r}, {self.val!r})'


class Lexer(object):
    def __init__(self, s: str) -> None:
        self.tokens = []
        self.i = 0
        self.s = s

    def execute(self) -> list[token]:
        while self.i < len(self.s):
            if self.s[self.i].isspace():
                self.evaluate_space()
            elif self.s[self.i].isalpha():
                self.evaluate_alpha()
            elif self.s[self.i].isdigit():
                self.evaluate_digit()
            elif self.s[self.i:self.i+2] in boolean_symbols:
                self.evaluate_boolean_symbol()
            elif self.s[self.i] == '!':
                self.evaluate_symbol()
            elif self.s[self.i:self.i+2] in double_len_symbols:
                self.evaluate_double_symbol()
            elif self.s[self.i] in single_len_symbols:
                self.evaluate_symbol()
            else:
                raise SyntaxError(f'unexpected character {self.s[self.i]}')

        return [token for token in self.tokens if token.typ != 'space']

    def evaluate_space(self):
        """
        >>> lexer = Lexer('3 + 4')
        >>> lexer.execute()
        [token('fl', '3'), token('sym', '+'), token('fl', '4')]
        >>> lexer.tokens
        [token('fl', '3'), token('space', ' '), token('sym', '+'), token('space', ' '), token('fl', '4')]
        """
        self.tokens.append(token('space', " "))
        self.i += 1

    def evaluate_alpha(self):
        end = self.i + 1
        while end < len(self.s) and (self.s[end].isalnum() or self.s[end] == '_'):
            end += 1
        assert end >= len(self.s) or not (
            self.s[end].isalnum() or self.s[end] == '_')

        word = self.s[self.i:end]

        if word in keywords:
            self.tokens.append(token('kw', word))
        else:
            self.tokens.append(token('var', word))

        self.i = end

    def evaluate_digit(self):
        end = self.i + 1
        while end < len(self.s) and (self.s[end] == '.' or self.s[end].isdigit()):
            end += 1
        assert end >= len(self.s) or not (
            self.s[end] == '.' or self.s[end].isdigit())

        self.tokens.append(token('fl', self.s[self.i:end]))

        self.i = end

    def evaluate_boolean_symbol(self):
        self.tokens.append(token('sym', self.s[self.i:self.i+2]))
        self.i += 2

    def evaluate_symbol(self):
        self.tokens.append(token('sym', self.s[self.i]))
        self.i += 1

    def evaluate_double_symbol(self):
        if (self.s[self.i:self.i+2] == '++' or self.s[self.i:self.i+2] == '--'):
            symbol = self.s[self.i:self.i+2]
            if self.tokens and self.tokens[-1].typ == 'var':
                # If the previous token is a variable, this is a post-sym operator
                self.tokens.append(token('sym', symbol))
                self.i += 2
            elif not self.tokens or self.tokens[-1].typ == 'space' or self.tokens[-1].val not in ["-", "+", '--', '++']:
                # If the previous token is a space, we can safely assume the next token is a variable
                self.tokens.append(token('sym', symbol))
                self.i += 2
            else:
                # Otherwise, we raise error?
                raise SyntaxError(
                    f'unexpected symbol {self.s[self.i:self.i+2]}'
                )
            return
        self.tokens.append(token('sym', self.s[self.i:self.i+2]))
        self.i += 2


class ast():
    typ: str
    children: tuple[Any, ...]
    post_op: Any

    def __init__(self, typ: str, *children: Any):
        self.typ = typ
        self.children = children
        self.post_op = None

    def __repr__(self):
        return f'ast({self.typ!r}, {", ".join([repr(c) for c in self.children])})'

    def add_post_op(self, post_op: Any):
        self.post_op = post_op


class Parsor(object):
    def __init__(self, s) -> None:
        self.s = s
        self.ts = []

    def execute(self):
        self.ts = Lexer(self.s).execute()

        a, i = self.bool_and_or(0)

        if i != len(self.ts):
            raise SyntaxError(f"expected EOF, found {self.ts[i:]!r}")

        return a

    def bool_and_or(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('x && y').execute()
        ast('&&', ast('var', 'x'), ast('var', 'y'))
        >>> Parsor('!x').execute()
        ast('!', ast('var', 'x'))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected boolean, found EOF')

        lhs, i = self.boolean_neg(i)

        while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in boolean_symbols:
            val = self.ts[i].val
            rhs, i = self.boolean_neg(i+1)
            lhs = ast(val, lhs, rhs)

        return lhs, i

    def boolean_neg(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('!x').execute()
        ast('!', ast('var', 'x'))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected boolean, found EOF')

        if self.ts[i].typ == 'sym' and self.ts[i].val == '!':
            a, i = self.boolean_neg(i+1)
            return ast('!', a), i
        else:
            return self.relational(i)

    def relational(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('x < y').execute()
        ast('<', ast('var', 'x'), ast('var', 'y'))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected relational, found EOF')

        lhs, i = self.plus_or_minus(i)

        while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in relational_symbols:
            val = self.ts[i].val
            rhs, i = self.plus_or_minus(i+1)
            lhs = ast(val, lhs, rhs)

        return lhs, i

    def plus_or_minus(self, i: int) -> tuple[ast, int]:
        if i >= len(self.ts):
            raise SyntaxError('expected plus_or_minus, found EOF')

        lhs, i = self.mul_or_div(i)

        while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in disj_symbols:
            val = self.ts[i].val
            rhs, i = self.mul_or_div(i+1)
            lhs = ast(val, lhs, rhs)

        return lhs, i

    def mul_or_div(self, i: int) -> tuple[ast, int]:
        if i >= len(self.ts):
            raise SyntaxError('expected mul_or_div, found EOF')

        lhs, i = self.power(i)

        while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in conj_symbols:
            val = self.ts[i].val
            rhs, i = self.power(i+1)
            lhs = ast(val, lhs, rhs)

        return lhs, i

    # Right associative power conjunction

    def power(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('2 ^ 3').execute()
        ast('^', ast('fl', 2.0), ast('fl', 3.0))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected power conjunction, found EOF')

        lhs, i = self.neg(i)

        if (i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val == '^'):
            rhs, i = self.power(i+1)
            lhs = ast('^', lhs, rhs)

        return lhs, i

    def neg(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('-1').execute()
        ast('-', ast('fl', 1.0))
        """

        if i >= len(self.ts):
            raise SyntaxError('expected negation, found EOF')

        if self.ts[i].typ == 'sym' and self.ts[i].val in neg_symbols:
            val = self.ts[i].val
            a, i = self.neg(i+1)
            return ast(val, a), i
        else:
            return self.incr_and_decr(i)

    def incr_and_decr(self, i: int) -> tuple[ast, int]:
        # pre and post incr and decr implemented
        # They are non-associative

        if i >= len(self.ts):
            raise SyntaxError('expected incr/decr, found EOF')

        if self.ts[i].typ == 'sym' and self.ts[i].val in incr_or_decr_symbols:
            val = self.ts[i].val
            if self.ts[i+1].typ != 'var':
                raise SyntaxError(
                    f'expected variable, found {self.ts[i+1].typ}'
                )
            a, i = self.atom(i+1)
            return ast(val, a), i

        if self.ts[i].typ == 'var' and i+1 < len(self.ts) and self.ts[i+1].typ == 'sym' and self.ts[i+1].val in incr_or_decr_symbols:
            op = self.ts[i+1].val
            var = self.ts[i].val
            ast_node = ast('var', var)
            ast_node.add_post_op(ast(op, ast('var', var)))
            return ast_node, i+2

        return self.atom(i)

    def atom(self, i: int) -> tuple[ast, int]:
        if i >= len(self.ts):
            raise SyntaxError('expected negation, found EOF')

        t = self.ts[i]

        if t.typ == 'var':
            VariableNameChecker(t.val).validate()
            return ast('var', t.val), i+1
        elif t.typ == 'fl':
            return ast('fl', float(t.val)), i+1
        elif t.typ == 'sym' and t.val == '(':
            a, i = self.bool_and_or(i + 1)

            if i >= len(self.ts):
                raise SyntaxError(f'expected right paren, got EOF')

            if not (self.ts[i].typ == 'sym' and self.ts[i].val == ')'):
                raise SyntaxError(f'expected right paren, got "{self.ts[i]}"')

            return a, i + 1

        raise SyntaxError(f'expected atom, got "{self.ts[i]}"')


class Interpreter(object):
    """
    >>> Interpreter(Parsor('x + y * z').execute(), {'y': 2, 'z': 3}).execute()
    6.0
    >>> Interpreter(Parsor('x / y * z').execute(), {'y': 2, 'z': 3}).execute()
    0.0
    >>> Interpreter(Parsor('x / y * z').execute(), {'z': 3, 'x': 6}).execute()
    Traceback (most recent call last):
    ...
    ZeroDivisionError: divide by zero
    >>> Interpreter(Parsor('(x + y) * z + 5').execute(), {'x': 6, 'y': 2, 'z': 3}).execute()
    29.0
    >>> Interpreter(Parsor('(x + y) * (z + 5)').execute(), {'x': 6, 'y': 2, 'z': 3}).execute()
    64.0
    >>> Interpreter(Parsor('(x + y) * (z - 5)').execute(), {'x': 6, 'y': 2, 'z': 3}).execute()
    -16.0
    """

    def __init__(self, a: ast, variables) -> None:
        self.a = a
        self.variables = variables

    def execute(self) -> bool:
        if self.a.typ in ['fl', 'bool']:
            return self.a.children[0]
        elif self.a.typ == 'var':
            return self.interp_var()
        elif self.a.typ == '=':
            return self.interp_assign()
        elif self.a.typ == '+':
            return self.interp_plus()
        elif self.a.typ == '-':
            return self.interp_minus()
        elif self.a.typ == '*':
            return self.interp_times()
        elif self.a.typ == '/':
            return self.interp_divide()
        elif self.a.typ == '%':
            return self.interp_mod()
        elif self.a.typ == '^':
            return self.interp_pow()
        elif self.a.typ == '==':
            return self.interp_eq()
        elif self.a.typ == '!=':
            return self.interp_neq()
        elif self.a.typ == '>':
            return self.interp_gt()
        elif self.a.typ == '<':
            return self.interp_lt()
        elif self.a.typ == '>=':
            return self.interp_gte()
        elif self.a.typ == '<=':
            return self.interp_lte()
        elif self.a.typ == '!':
            return self.interp_not()
        elif self.a.typ == '&&':
            return self.interp_and()
        elif self.a.typ == '||':
            return self.interp_or()
        elif self.a.typ in ['--', '++']:
            return self.interp_incr_or_decr()

        raise SyntaxError(f'unknown operation {self.a.typ}')

    def interp_var(self):
        if self.a.children[0] not in self.variables:
            self.variables[self.a.children[0]] = float(0)

        value = self.variables[self.a.children[0]]
        if self.a.post_op:
            Interpreter(self.a.post_op, self.variables).execute()
        return value

    def interp_plus(self):
        return Interpreter(self.a.children[0], self.variables).execute() + Interpreter(self.a.children[1], self.variables).execute()

    def interp_minus(self):
        if len(self.a.children) == 1:
            return -Interpreter(self.a.children[0], self.variables).execute()

        return Interpreter(self.a.children[0], self.variables).execute() - Interpreter(self.a.children[1], self.variables).execute()

    def interp_times(self):
        return Interpreter(self.a.children[0], self.variables).execute() * Interpreter(self.a.children[1], self.variables).execute()

    def interp_divide(self):
        right = Interpreter(self.a.children[1], self.variables).execute()
        if right in [0, None, 0.0]:
            raise ZeroDivisionError('divide by zero')
        return Interpreter(self.a.children[0], self.variables).execute() / right

    def interp_mod(self):
        right = Interpreter(self.a.children[1], self.variables).execute()
        if right in [0, None, 0.0]:
            raise ZeroDivisionError('divide by zero')
        left = Interpreter(self.a.children[0], self.variables).execute()

        return left - (right * int(left/right))
        # return left % right

    def interp_pow(self):
        return Interpreter(self.a.children[0], self.variables).execute() ** Interpreter(self.a.children[1], self.variables).execute()

    def interp_not(self):
        return self.bool_to_int(not Interpreter(self.a.children[0], self.variables).execute())

    def interp_and(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() and Interpreter(self.a.children[1], self.variables).execute())

    def interp_or(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() or Interpreter(self.a.children[1], self.variables).execute())

    def interp_incr_or_decr(self):
        if len(self.a.children) != 1:
            raise SyntaxError(f'expected 1 child, got {len(self.a.children)}')

        if self.a.children[0].typ != 'var':
            raise SyntaxError(
                f'expected variable, got {self.a.children[0].typ}'
            )

        variable = self.a.children[0].children[0]

        if variable not in self.variables:
            self.variables[variable] = float(0)

        if self.a.typ == '++':
            self.variables[variable] += 1
        elif self.a.typ == '--':
            self.variables[variable] -= 1
        else:
            raise SyntaxError(f'unknown operation {self.a.children[1]}')

        return self.variables[variable]

    def interp_eq(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() == Interpreter(self.a.children[1], self.variables).execute())

    def interp_neq(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() != Interpreter(self.a.children[1], self.variables).execute())

    def interp_gt(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() > Interpreter(self.a.children[1], self.variables).execute())

    def interp_lt(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() < Interpreter(self.a.children[1], self.variables).execute())

    def interp_gte(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() >= Interpreter(self.a.children[1], self.variables).execute())

    def interp_lte(self):
        return self.bool_to_int(Interpreter(self.a.children[0], self.variables).execute() <= Interpreter(self.a.children[1], self.variables).execute())

    def bool_to_int(self, b):
        return 1 if b else 0


class StatementEvaluator(object):
    def __init__(self, statements):
        self.statements = statements
        self.variables = {}
        self.parsed_statements = []
        self.parse_paused_statement = None
        self.printlist = []
        self.in_block_comment = False

    def execute(self):
        try:
            self.parse()
        except (SyntaxError, ValueError):
            print("parse error")
            return

        try:
            self.evaluate()
        except ZeroDivisionError:
            print(*(self.printlist + ["divide by zero"]))
            return

    def parse(self):
        for statement in self.statements:
            self.parse_statement(statement)

    def parse_statement(self, statement):
        statement = statement.strip()
        if not statement:
            return

        parsed_statement = StatementParser(
            statement, self.in_block_comment
        ).parse()
        if parsed_statement.get('block_comment', False):
            self.in_block_comment = True
            if not self.parse_paused_statement:
                self.parse_paused_statement = ''
            self.parse_paused_statement += parsed_statement['statement']
            return
        else:
            if self.in_block_comment:
                self.in_block_comment = False
                self.parse_paused_statement += parsed_statement['statement']
                # Start parsing again from the beginning
                self.parse_statement(self.parse_paused_statement)
                # Reset the paused statement
                self.parse_paused_statement = None
                return

        self.parsed_statements.append(parsed_statement)

    def evaluate(self):
        """
        Doctests for all operators
        
        """
        if not self.parsed_statements:
            return

        for statement in self.parsed_statements:
            if statement['type'] == 'print':
                self.printlist = []
                if not statement['value']:
                    print()
                    continue

                for item in statement['value']:
                    if isinstance(item, str):
                        self.printlist.append(item)
                    else:
                        val = Interpreter(item, self.variables)
                        result = val.execute()
                        self.printlist.append(result)
                print(*self.printlist, sep=' ')
                self.printlist = []
            elif statement['type'] == 'assign':
                self.variables[statement['variable']] = Interpreter(
                    statement['value'], self.variables
                ).execute()
            else:
                Interpreter(
                    statement['value'], self.variables
                ).execute()


class StatementParser(object):
    def __init__(self, statement, block_comment):
        self.statement = statement
        self.index = 0
        self.block_comment = block_comment
        self.block_comment_ended = False

    def parse(self):
        # Remove commented code from statement
        self.sanitize_statement()

        if self.block_comment:
            return {'statement': self.statement, 'block_comment': True}
        
        if self.block_comment_ended:
            return {'statement': self.statement, 'block_comment': False}

        while self.index < len(self.statement):
            if self.index == 0 and self.statement.startswith('print'):
                if len(self.statement) > 5 and self.statement[5] != ' ':
                    self.index = 5
                    continue
                linestatement = self.statement[6:].strip()
                linestatement = linestatement.split(',')
                if not linestatement:
                    return self.print_eval_dict_builder('print', [])
                else:
                    printlist = [Parsor(i).execute() for i in linestatement]
                    return self.print_eval_dict_builder('print', printlist)

            if self.statement[self.index] == ' ':
                self.index += 1
                continue

            if self.statement[self.index:self.index + 2] in op_equals_symbols:
                return self.parse_op_equate()

            if self.statement[self.index:self.index + 3] in bool_equals_symbols:
                return self.parse_op_bool_equate()

            if self.statement[self.index] == '=':
                return self.parse_equate()

            self.index += 1

        self.statement = self.statement.strip()

        if self.statement:
            return self.print_eval_dict_builder('eval', Parsor(self.statement).execute())

    def sanitize_statement(self):
        if self.block_comment:
            comment_type = '*/'
            comment_idx = self.find_comment_type(self.statement, comment_type)

            if comment_idx is None:
                self.statement = ''
                return
        else:
            comment_type, comment_idx = self.find_any_comment(
                self.statement
            )

        if not comment_type:
            return True

        if comment_type == '#':
            self.statement = self.statement[:comment_idx]
            return True

        if comment_type == '/*':
            new_statement = self.statement[:comment_idx]
            end_comment_idx = self.find_comment_type(
                self.statement[comment_idx:], '*/'
            )
            if end_comment_idx:
                new_statement += self.statement[
                    comment_idx + end_comment_idx + 2:
                ]
                self.statement = new_statement
                self.block_comment = False
                self.block_comment_ended = True
                # Calling sanitize statement again to check for any other comments
                self.sanitize_statement()
            else:
                self.statement = new_statement
                self.block_comment = True
                self.block_comment_ended = False

        if comment_type == '*/':
            if not self.block_comment:
                raise SyntaxError('Unexpected */')
            
            new_statement = self.statement[comment_idx + 2:]
            self.statement = new_statement
            self.block_comment = False
            self.block_comment_ended = True
            # Calling sanitize statement again to check for any other comments
            self.sanitize_statement()


    def find_any_comment(self, statement):
        comment_idxs = [
            statement.index(i)
            for i in ('#', '/*', '*/')
            if i in statement
        ]
        idx = min(comment_idxs) if len(comment_idxs) > 0 else None

        if idx is None:
            return (None, statement)

        return ('#' if statement[idx] == '#' else statement[idx:idx+2], idx)

    def find_comment_type(self, statement, type):
        if not type or not statement:
            return None

        if type == '#':
            return statement.index('#') if '#' in statement else None

        if type == '/*':
            return statement.index('/*') if '/*' in statement else None

        if type == '*/':
            return statement.index('*/') if '*/' in statement else None

        return None

    def parse_equate(self):
        variable = self.statement[:self.index].strip()
        expression = self.statement[self.index+1:].strip()
        if not expression or not variable:
            raise SyntaxError('parse error')

        VariableNameChecker(variable).validate()

        return self.assign_dict_builder(
            variable,
            Parsor(expression).execute()
        )

    def parse_op_equate(self):
        if self.statement[self.index:self.index+2] not in op_equals_symbols:
            raise SyntaxError('parse error')
        return self.parse_op_equate_helper(2)

    def parse_op_bool_equate(self):
        if self.statement[self.index:self.index+3] not in bool_equals_symbols:
            raise SyntaxError('parse error')
        return self.parse_op_equate_helper(3)

    def parse_op_equate_helper(self, length):
        variable = self.statement[:self.index].strip()
        operation = self.statement[self.index:self.index+length]
        expression = self.statement[self.index+length:].strip()
        if not expression or not variable:
            raise SyntaxError('parse error')

        VariableNameChecker(variable).validate()

        return self.assign_dict_builder(
            variable,
            Parsor(
                f'{variable}{operation.replace("=", "")}{expression}'
            ).execute()
        )

    def assign_dict_builder(self, key, value):
        return {
            'type': 'assign',
            'variable': key,
            'value': value
        }

    def print_eval_dict_builder(self, typ, value):
        return {
            'type': typ,
            'value': value
        }


class VariableNameChecker(object):
    def __init__(self, name):
        self.name = name

    def validate(self):
        if self.not_matched_with_regex() or self.all_underscore() or self.not_keyword():
            raise SyntaxError('parse error')

    def not_matched_with_regex(self):
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.name) is None

    def all_underscore(self):
        return all([i == '_' for i in self.name])

    def not_keyword(self):
        return self.name in keywords


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    statements = []
    for line in sys.stdin:
        if line:
            statements.append(line.strip())
    StatementEvaluator(statements).execute()
