class Parser(object):
    def __init__(self, expression):
        self.expression = expression
    def execute(self):
        """
        This method parses given expression and returns list of items
        """
        self.expression = self.expression.split('\n')
        return self.expression

    # More functions to follow


class Evaluator(object):
    def __init__(self, items, variables):
        self.items = items
        self.variables = variables
    def execute(self):
        """
        This method evaluates given list of items and returns result
        """
        # print(self.items)
        for statements in self.items:
          if statements.startswith("print "):
            expression = statements[6: ].strip()
            val = eval(expression, self.variables)
            return val
          else:
            statements = statements.replace(' ', '')
            calculate = statements.split('=')
            variable = calculate[0]
            expression = calculate[1]
            try:
              val = eval(expression, self.variables)
            except NameError:
              return "NameError"
            self.variables[variable] = val
        pass

    # More functions to follow


class ExpressionEvaluator(object):
    def __init__(self, expression):
        self.expression = expression
        self.items = None
        self.variables = dict()

    def execute(self):
        """
        This method parses given expression, evaluates it and returns the final result
        """
        parsedlist = self.expression.split('\n')
        return parsedlist

    # More functions to follow


if __name__ == '__main__':
  lines = ""
  while True:
    line = input()
    if line:
      lines += line + '\n'
    else:
      break
  parser = Parser(lines)
  evaluator = Evaluator(parser.execute(), {})
  print(evaluator.execute())
