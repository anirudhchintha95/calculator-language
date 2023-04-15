from typing import Any
import sys


single_len_symbols = ["-", "+", "*", "/", "%", "^", "!", "(", ")", "<", ">"]
double_len_symbols = ["||", "&&", "++", "--", "==", "!=", "<=", ">="]

disj_symbols = ["+", "-", "||"]
conj_symbols = ["*", "/", "%", "&&"]
power_conj_symbols = ["^"]
neg_symbols = ["!", "-"]
other_symbols = ["--", "++"]


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
            elif self.s[self.i:self.i+2] in double_len_symbols:
                self.evaluate_double_symbol()
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

    def evaluate_symbol(self):
        self.tokens.append(token('sym', self.s[self.i]))
        self.i += 1

    def evaluate_double_symbol(self):
        self.tokens.append(token('sym', self.s[self.i:self.i+2]))
        self.i += 2


class ast():
    typ: str
    children: tuple[Any, ...]

    def __init__(self, typ: str, *children: Any):
        """
        x || true
        >>> ast('||', ast('var', 'x'), ast('bool', True))
        ast('||', ast('var', 'x'), ast('bool', True))
        """
        self.typ = typ
        self.children = children

    def __repr__(self):
        return f'ast({self.typ!r}, {", ".join([repr(c) for c in self.children])})'


class Parsor(object):
    def __init__(self, s) -> None:
        self.s = s
        self.ts = []

    def execute(self):
        self.ts = Lexer(self.s).execute()

        a, i = self.disj(0)

        if i != len(self.ts):
            raise SyntaxError(f"expected EOF, found {self.ts[i:]!r}")

        return a

    def disj(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('true || false').execute()
        ast('||', ast('bool', True), ast('bool', False))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected disjunction, found EOF')

        lhs, i = self.conj(i)

        while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in disj_symbols:
            val = self.ts[i].val
            rhs, i = self.conj(i+1)
            lhs = ast(val, lhs, rhs)

        return lhs, i

    def conj(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('true && false').execute()
        ast('&&', ast('bool', True), ast('bool', False))
        >>> Parsor('!x && (a && !false)').execute()
        ast('&&', ast('!', ast('var', 'x')), ast('&&', ast('var', 'a'), ast('!', ast('bool', False))))
        >>> Parsor('!x && a && !false').execute()
        ast('&&', ast('&&', ast('!', ast('var', 'x')), ast('var', 'a')), ast('!', ast('bool', False)))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected conjunction, found EOF')

        lhs, i = self.power_conj(i)

        while i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val in conj_symbols:
            val = self.ts[i].val
            rhs, i = self.power_conj(i+1)
            lhs = ast(val, lhs, rhs)

        return lhs, i

    # Right associative power conjunction

    def power_conj(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('2 ^ 3').execute()
        ast('^', ast('fl', 2.0), ast('fl', 3.0))
        """
        if i >= len(self.ts):
            raise SyntaxError('expected power conjunction, found EOF')

        lhs, i = self.neg(i)

        if (i < len(self.ts) and self.ts[i].typ == 'sym' and self.ts[i].val == '^'):
            rhs, i = self.power_conj(i+1)
            lhs = ast('^', lhs, rhs)

        return lhs, i

    def neg(self, i: int) -> tuple[ast, int]:
        """
        >>> Parsor('! true').execute()
        ast('!', ast('bool', True))
        >>> Parsor('!! true').execute()
        ast('!', ast('!', ast('bool', True)))
        """

        if i >= len(self.ts):
            raise SyntaxError('expected negation, found EOF')

        if self.ts[i].typ == 'sym' and self.ts[i].val in neg_symbols:
            val = self.ts[i].val
            a, i = self.neg(i+1)
            return ast(val, a), i
        else:
            # return self.incr_and_decr(i)
            return self.atom(i)

    # TODO: Fix this

    def incr_and_decr(self, i: int) -> tuple[ast, int]:
        pass
    #     """
    #     >>> Parsor('++x').execute()
    #     ast('++', ast('var', 'x'))
    #     >>> Parsor('x++').execute()
    #     ast('++', ast('var', 'x'))
    #     >>> Parsor('++x++').execute()
    #     ast('++', ast('++', ast('var', 'x')))
    #     """

    #     if i >= len(self.ts):
    #         raise SyntaxError('expected increment, found EOF')

    #     if self.ts[i].typ == 'sym' and self.ts[i].val in other_symbols:
    #         val = self.ts[i].val
    #         a, i = self.incr_and_decr(i+1)
    #         return ast(val, a), i
    #     else:
    #         return self.atom(i)

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
            return ast('var', t.val), i+1
        elif t.typ == 'fl':
            return ast('fl', float(t.val)), i+1
        elif t.typ == 'kw' and t.val in ['true', 'false']:
            return ast('bool', t.val == 'true'), i + 1
        elif t.typ == 'sym' and t.val == '(':
            a, i = self.disj(i + 1)

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
    ZeroDivisionError: division by zero
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
        elif self.a.typ == '!':
            return not self.interp_not()
        elif self.a.typ == '&&':
            return self.interp_and()
        elif self.a.typ == '||':
            return self.interp_or()

        # TODO: Handle other operations ++, --, etc.

        raise SyntaxError(f'unknown operation {self.a.typ}')

    def interp_var(self):
        return self.variables[self.a.children[0]] if self.a.children[0] in self.variables else float(0)

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
            raise ZeroDivisionError('division by zero')
        return Interpreter(self.a.children[0], self.variables).execute() / right

    def interp_mod(self):
        right = Interpreter(self.a.children[1], self.variables).execute()
        if right in [0, None, 0.0]:
            raise ZeroDivisionError('division by zero')
        return Interpreter(self.a.children[0], self.variables).execute() % right

    def interp_pow(self):
        return Interpreter(self.a.children[0], self.variables).execute() ** Interpreter(self.a.children[1], self.variables).execute()

    def interp_not(self):
        return not Interpreter(self.a.children[0], self.variables).execute()

    def interp_and(self):
        return Interpreter(self.a.children[0], self.variables).execute() and Interpreter(self.a.children[1], self.variables).execute()

    def interp_or(self):
        return Interpreter(self.a.children[0], self.variables).execute() or Interpreter(self.a.children[1], self.variables).execute()

    def interp_incr_decr(self):
        pass


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
        except SyntaxError:
            print("parse error")
            return

        try:
            self.evaluate()
        except ZeroDivisionError:
            print(*(self.printlist + ["division by zero"]))
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

        statement = statement.strip().replace(' ', '')

        # TODO: Fill these up
        if '+=' in statement:
          calculate = statement.split('+=')
          variable = calculate[0]
          expression = f'{variable}+{calculate[1]}'
          self.Parse_evaluate(expression, variable)
        elif '-=' in statement:
          calculate = statement.split('-=')
          variable = calculate[0]
          expression = f'{variable}-{calculate[1]}'
          self.Parse_evaluate(expression, variable)
        elif '*=' in statement:
          calculate = statement.split('*=')
          variable = calculate[0]
          expression = f'{variable}*{calculate[1]}'
          self.Parse_evaluate(expression, variable)
        elif '/=' in statement:
          calculate = statement.split('/=')
          variable = calculate[0]
          expression = f'{variable}/{calculate[1]}'
          self.Parse_evaluate(expression, variable)
        elif '%=' in statement:
          calculate = statement.split('%=')
          variable = calculate[0]
          expression = f'{variable}%{calculate[1]}'
          self.Parse_evaluate(expression, variable)
        elif '^=' in statement:
          calculate = statement.split('^=')
          variable = calculate[0]
          expression = f'{variable}^{calculate[1]}'
          self.Parse_evaluate(expression, variable)
        elif '=' in statement:
            calculate = statement.split('=')
            variable = calculate[0]
            expression = calculate[1]
            self.Parse_evaluate(expression, variable)
        elif '++' in statement:
            return
        elif '--' in statement:
            return
        elif any([i in statement for i in single_len_symbols]) or any([i in statement for i in single_len_symbols]):
            self.parsed_statements.append({
                'type': 'eval',
                'value': Parsor(statement).execute()
            })
        else:
            self.parsed_statements.append({
                'type': 'assign',
                'variable': statement,
                'value': Parsor('0').execute()
            })

    def Parse_evaluate(self, expression, variable):
      if not expression or not variable:
                raise SyntaxError('parse error')

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
            elif type == 'eval':
                Interpreter(statement['value'], self.variables).execute()
            elif statement['type'] == 'assign':
                self.variables[statement['variable']] = Interpreter(
                    statement['value'], self.variables
                ).execute()


if __name__ == '__main__':
    # lines = ""
    # while True:
    #     line = input()
    #     if line:
    #         lines += line + '\n'
    #     else:
    #         break
    # StatementEvaluator(lines.split('\n')).execute()
    statements = []
    for line in sys.stdin:
        if line:
            statements.append(line.strip())
    StatementEvaluator(statements).execute()
