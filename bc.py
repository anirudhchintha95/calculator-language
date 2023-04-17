from typing import Any
import re
import sys


single_len_symbols = ["-", "+", "*", "/", "%", "^", "(", ")", "<", ">"]
boolean_symbols = ["&&", "||", "!"]
double_len_symbols = ["++", "--", "==", "!=", "<=", ">="]
op_equals_symbols = ["+=", "-=", "*=", "/=", "%=", "^=", "&&=", "||=", "!="]
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
            elif self.s[self.i:self.i+2] in op_equals_symbols:
                self.evaluate_double_symbol()
            elif self.s[self.i:self.i+2] in double_len_symbols:
                self.evaluate_double_symbol()
            elif self.s[self.i] in assign_symbols:
                self.evaluate_symbol()
            elif self.s[self.i] in single_len_symbols:
                self.evaluate_symbol()
            else:
                raise SyntaxError(f'unexpected character {self.s[self.i]}')

        return self.tokens

    def evaluate_space(self):
        self.i += 1

    def evaluate_alpha(self):
        end = self.i + 1
        while end < len(self.s) and (self.s[end].isalnum() or self.s[end] == '_'):
            end += 1
        assert end >= len(self.s) or not (
            self.s[end].isalnum() or self.s[end] == '_')

        word = self.s[self.i:end]

        if word in ['true', 'false']:
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
            elif not self.tokens or self.tokens[-1].val in single_len_symbols or (self.tokens[-1].val in double_len_symbols and self.tokens[-1].val != '++' and self.tokens[-1].val != '--'):
                # If the previous token is a symbol or there is no previous token, this is a pre-sym operator
                self.tokens.append(token('sym', symbol))
                self.i += 2
            else:
                # Otherwise, this is a regular single length symbol
                # Or should we raise error?
                self.evaluate_symbol()
                # raise SyntaxError(
                #     f'unexpected symbol {self.s[self.i:self.i+2]}')
            return
        self.tokens.append(token('sym', self.s[self.i:self.i+2]))
        self.i += 2


class ast():
    typ: str
    children: tuple[Any, ...]
    post_op: Any

    def __init__(self, typ: str, *children: Any):
        """
        x || true
        >>> ast('||', ast('var', 'x'), ast('bool', True))
        ast('||', ast('var', 'x'), ast('bool', True))
        """
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
        self.i = 0

    def execute(self):
        self.ts = Lexer(self.s).execute()
        a, i = self.assign(0)
        if i != len(self.ts):
            raise SyntaxError(f"expected EOF, found {self.ts[i:]!r}")

        return a

    def assign(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('x = y').execute()
        ast('=', ast('var', 'x'), ast('var', 'y'))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected assign, found EOF')

        lhs, i = self.parse_logic(i)
        # lhs, i = self.parse_logic(i)
        if i < len(self.ts) and self.ts[i].typ == 'sym':
            if self.ts[i].val in op_equals_symbols:
                if lhs.typ != 'var':
                    raise SyntaxError(
                        'expected variable, found ' + lhs.typ
                    )
                val = self.ts[i].val
                # rhs, i = self.relational(i+1)
                rhs, i = self.parse_logic(i+1)
                if i != len(self.ts):
                    raise SyntaxError(f"expected EOF, found {self.ts[i:]!r}")
                lhs = ast('=', lhs, ast(val.replace('=', ''), lhs, rhs))
                print(lhs)
            elif self.ts[i].val in assign_symbols:
                while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in assign_symbols:
                    if not (lhs.typ == 'var' or lhs.typ == '='):
                        raise SyntaxError(
                            'expected variable, found ' + lhs.typ)
                    val = self.ts[i].val
                    rhs, i = self.parse_logic(i+1)
                    # rhs, i = self.parse_logic(i+1)
                    lhs = ast(val, lhs, rhs)

        return lhs, i
    
    def parse_logic(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('x && y').execute()
        ast('&&', ast('var', 'x'), ast('var', 'y'))
        >>> Parsor('!x').execute()
        ast('!', ast('var', 'x'))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected boolean, found EOF')
        
        # lhs, i = self.plus_or_minus(i)
        if self.ts[i].val != '!':
            lhs, i = self.relational(i)

        while i< len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in boolean_symbols:
            if self.ts[i].val != '!':
                val = self.ts[i].val
                rhs, i = self.relational(i+1)
                lhs = ast(val, lhs, rhs)
            
            elif self.ts[i].typ == 'sym' and self.ts[i].val == '!':
                val = self.ts[i].val
                a, i = self.parse_logic(i+1)
                return ast(val, a), i

        return lhs, i

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
        """
        >>> Parsor('true || false').execute()
        ast('||', ast('bool', True), ast('bool', False))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected plus_or_minus, found EOF')

        lhs, i = self.mul_or_div(i)

        while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in disj_symbols:
            val = self.ts[i].val
            rhs, i = self.mul_or_div(i+1)
            lhs = ast(val, lhs, rhs)

        return lhs, i

    def mul_or_div(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('true && false').execute()
        ast('&&', ast('bool', True), ast('bool', False))
        >>> Parsor('!x && (a && !false)').execute()
        ast('&&', ast('!', ast('var', 'x')), ast(
            '&&', ast('var', 'a'), ast('!', ast('bool', False))))
        >>> Parsor('!x && a && !false').execute()
        ast('&&', ast('&&', ast('!', ast('var', 'x')), ast(
            'var', 'a')), ast('!', ast('bool', False)))
        """
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
        """
        >>> Parsor('x').execute()
        ast('var', 'x')
        >>> Parsor('true').execute()
        ast('bool', True)
        >>> Parsor('(((false)))').execute()
        ast('bool', False)
        """

        if i >= len(self.ts):
            raise SyntaxError('expected negation, found EOF')

        t = self.ts[i]

        if t.typ == 'var':
            self.check_var_name_validity(t.val)
            return ast('var', t.val), i+1
        elif t.typ == 'fl':
            return ast('fl', float(t.val)), i+1
        elif t.typ == 'kw' and t.val in ['true', 'false']:
            return ast('bool', t.val == 'true'), i + 1
        elif t.typ == 'sym' and t.val == '(':
            a, i = self.assign(i + 1)

            if i >= len(self.ts):
                raise SyntaxError(f'expected right paren, got EOF')

            if not (self.ts[i].typ == 'sym' and self.ts[i].val == ')'):
                raise SyntaxError(f'expected right paren, got "{self.ts[i]}"')

            return a, i + 1

        raise SyntaxError(f'expected atom, got "{self.ts[i]}"')

    def check_var_name_validity(self, name):
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is None or all([i == '_' for i in name]) or name in keywords:
            raise SyntaxError('parse error')


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
            if self.interp_and() == 0:
                return 0
            elif self.interp_and() > 0:
                return 1
        elif self.a.typ == '||':
            if self.interp_or() == 0:
                return 0
            elif self.interp_or() > 0:
                return 1
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

    def interp_assign(self):
        if not (self.a.children[0].typ == 'var' or self.a.children[0].typ == "="):
            raise SyntaxError(
                f'expected variable, got {self.a.children[0].typ}'
            )
        if self.a.children[0].typ == 'var':
            variable = self.a.children[0].children[0]
            self.variables[variable] = Interpreter(
                self.a.children[1], self.variables
            ).execute()
            return self.variables[variable]
        else:
            if self.a.children[0].children[1].typ != 'var':
                raise SyntaxError(
                    f'expected variable, got {self.a.children[0].children[1].typ}'
                )
            next_child_variable = self.a.children[0].children[1].children[0]
            self.variables[next_child_variable] = Interpreter(
                self.a.children[1], self.variables
            ).execute()
            return Interpreter(self.a.children[0], self.variables).execute()

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
        return not Interpreter(self.a.children[0], self.variables).execute()

    def interp_and(self):
        return Interpreter(self.a.children[0], self.variables).execute() and Interpreter(self.a.children[1], self.variables).execute()

    def interp_or(self):
        return Interpreter(self.a.children[0], self.variables).execute() or Interpreter(self.a.children[1], self.variables).execute()

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
        return int(b)


class StatementEvaluator(object):
    def __init__(self, statements):
        self.statements = statements
        self.variables = {}
        self.parsed_statements = []
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
        if self.in_block_comment:
            if '*/' in statement:
                self.in_block_comment = False
                self.parse_statement(statement.split('*/')[1])
                return

        if self.in_block_comment:
            return

        if not statement:
            return

        has_comment = '#' in statement
        has_block_comment = '/*' in statement

        if has_comment and has_block_comment:
            comment_indexes = (statement.index('/*'), statement.index('#'))
            if comment_indexes[0] < comment_indexes[1]:
                self.in_block_comment = True
                self.parse_statement(statement.split('/*')[0])
                statement = statement.split('/*')[1]
                if '*/' in statement:
                    self.in_block_comment = False
                    self.parse_statement(statement.split('*/')[1])
            else:
                self.parse_statement(statement.split('#')[0])
            return

        if has_block_comment:
            self.in_block_comment = True
            self.parse_statement(statement.split('/*')[0])
            statement = statement.split('/*')[1]
            if '*/' in statement:
                self.in_block_comment = False
                self.parse_statement(statement.split('*/')[1])
                return
            return

        if has_comment:
            self.parse_statement(statement.split('#')[0])
            return

        if '*/' in statement and not self.in_block_comment:
            raise SyntaxError("parse error")

        statement = statement.strip()
        if not statement:
            return

        if statement.startswith("print "):
            linestatement = statement[6:].strip().replace(' ', '')
            linestatement = linestatement.split(',')
            if not linestatement:
                self.parsed_statements.append({
                    'type': 'print',
                    'value': []
                })
            else:
                printlist = [Parsor(i).execute() for i in linestatement]
                self.parsed_statements.append({
                    'type': 'print',
                    'value': printlist
                })
            return

        self.parsed_statements.append({
            'type': 'eval',
            'value': Parsor(statement).execute()
        })

        # statement = statement.strip().replace(' ', '')

        # if '+=' in statement:
        #     calculate = statement.split('+=')
        #     variable = calculate[0]
        #     expression = f'{variable}+{calculate[1]}'
        #     self.parse_equate(expression, variable)
        # elif '-=' in statement:
        #     calculate = statement.split('-=')
        #     variable = calculate[0]
        #     expression = f'{variable}-{calculate[1]}'
        #     self.parse_equate(expression, variable)
        # elif '*=' in statement:
        #     calculate = statement.split('*=')
        #     variable = calculate[0]
        #     expression = f'{variable}*{calculate[1]}'
        #     self.parse_equate(expression, variable)
        # elif '/=' in statement:
        #     calculate = statement.split('/=')
        #     variable = calculate[0]
        #     expression = f'{variable}/{calculate[1]}'
        #     self.parse_equate(expression, variable)
        # elif '%=' in statement:
        #     calculate = statement.split('%=')
        #     variable = calculate[0]
        #     expression = f'{variable}%{calculate[1]}'
        #     self.parse_equate(expression, variable)
        # elif '^=' in statement:
        #     calculate = statement.split('^=')
        #     variable = calculate[0]
        #     expression = f'{variable}^{calculate[1]}'
        #     self.parse_equate(expression, variable)
        # # TODO: Need to fix this for >= and <=
        # elif '=' in statement and statement[statement.index('=') - 1] not in ['+', '-', '*', '/', '%', '^', '>', '<']:
        #     self.assignment_parser(statement)
        # elif any([i in statement for i in single_len_symbols]) or any([i in statement for i in double_len_symbols]) or all([i.isdigit() or i == '.' for i in statement]):
        #     self.parsed_statements.append({
        #         'type': 'eval',
        #         'value': Parsor(statement).execute()
        #     })
        # else:
        #     self.check_var_name_validity(statement)
        #     self.parsed_statements.append({
        #         'type': 'assign',
        #         'variable': statement,
        #         'value': Parsor('0').execute()
        #     })

    def parse_equate(self, expression, variable):
        if not expression or not variable:
            raise SyntaxError('parse error')

        self.check_var_name_validity(variable)

        parsed_expression = Parsor(expression).execute()
        self.parsed_statements.append({
            'type': 'assign',
            'variable': variable,
            'value': parsed_expression
        })

    def evaluate(self):
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

    # This only splits by = and not by ==, >=, <=, etc.
    # def assignment_parser(self, statement):
    #     # ignore >=, <=, ==, !=, etc.
    #     # Also it could have multiple = in the statement
    #     # TODO: Need to fix this for >=, <=, != and ==
    #     items = statement.split('=')
    #     if len(items) < 2:
    #         raise SyntaxError('parse error')

    #     for i in items[:-1]:
    #         self.parse_equate(items[-1], i)

    def check_var_name_validity(self, name):
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is None or all([i == '_' for i in name]) or name in keywords:
            raise SyntaxError('parse error')


if __name__ == '__main__':
    statements = []
    for line in sys.stdin:
        if line:
            statements.append(line.strip())
    StatementEvaluator(statements).execute()
