class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        iterate through the array and keep adding numbers to stack until you encounter a symbol
        once you do, pop the top of stack and perform the operation of the symbol on that 
        along with the result already established (None at first, so we'll pop two numbers from 
        stack for very first expression)
        """
        stack = [];
        firstOperand = None;
        for token in tokens:
            print('self.is_number(token)', self.is_number(token));
            print('firstOperand', firstOperand);
            if self.is_number(token):
                stack.append(int(token));
            else:
                if firstOperand is None:
                    firstOperand = stack.pop();
                    secondOperand = stack.pop();
                else:
                    secondOperand = stack.pop();
                firstOperand = self.performOperation(secondOperand, firstOperand, token);
        return firstOperand;
    
    def is_number(self, element):
        if element.isdigit():  # For positive integers
            return True
        elif element[0] == '-' and element[1:].isdigit():  # For negative integers
            return True
        elif element.isnumeric():  # For positive integers with additional characters (e.g., "+123")
            return True
        else:
            try:
                float(element)  # For floating-point numbers
                return True
            except ValueError:
                return False
    
    def performOperation(self, first, second, token):
        print('performing ', first, token, second);
        if token == '+':
            return first + second;
        if token == '-':
            return first - second;
        if token == '*':
            return first * second;
        if token == '/':
            return int(first / second);



solution = Solution();
tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
result = solution.evalRPN(tokens);
print(result);