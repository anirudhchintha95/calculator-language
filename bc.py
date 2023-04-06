class Parser(object):
    def __init__(self, expression):
        self.expression = expression

    def execute(self):
        """
        This method parses given expression and returns list of items
        """
        pass

    # More functions to follow


class Evaluator(object):
    def __init__(self, items, variables):
        self.items = items
        self.variables = variables

    def execute(self):
        """
        This method evaluates given list of items and returns result
        """
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
        pass

    # More functions to follow


if __name__ == '__main__':
    expression = '1 + 2'
    evaluator = ExpressionEvaluator(expression)
    print(evaluator.execute())
