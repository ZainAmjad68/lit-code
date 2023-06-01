class Solution(object):
    def isValid(self, s):
        stack = []
        symbolDict = {'(': ')', '[': ']', '{': '}'}
        """
        We iterate through each symbol in the string.
        If the symbol is an opening symbol, we push it onto the stack.
        If the symbol is a closing symbol, we check if the stack is not empty and the top symbol on the stack has a matching closing symbol in symbolDict. If it does, we pop the top symbol from the stack.
        If the symbol is not a valid closing symbol or there is a mismatch between the opening and closing symbols, we return False.
        Finally, we check if the stack is empty after traversing the entire string. If it is, then all opening symbols have been closed properly, and we return True. Otherwise, we return False.
        """
        for symbol in s:
            print('stack', stack);
            if symbol in symbolDict:
                print('appending');
                stack.append(symbol)
            elif stack and symbolDict[stack[-1]] == symbol:
                print('popping');
                stack.pop()
            else:
                print('falsified');
                return False

        return len(stack) == 0
    
solution = Solution();
s = "({})[({})]{}";
result = solution.isValid(s);
print(result);




"""
attempt to use a hash map:
class Solution(object):
    def isValid(self, s):
        symbolDict = {};
        print('s', s);
        for symbol in s:
            print('symbol',symbol);
            if symbol in symbolDict:
                symbolDict[symbol] += 1;
            else:
                symbolDict[symbol] = 1;

            print('symbolDict', symbolDict);
        
        if (('(' in symbolDict and ')' not in symbolDict) or (')' in symbolDict and '(' not in symbolDict) or (symbolDict['('] != symbolDict[')'])):
            return False;

        if (('[' in symbolDict and ']' not in symbolDict) or (']' in symbolDict and '[' not in symbolDict) or (symbolDict['['] != symbolDict[']'])):
            return False;

        if (('{' in symbolDict and '}' not in symbolDict) or ('}' in symbolDict and '{' not in symbolDict) or (symbolDict['{'] != symbolDict['}'])):
            return False;
        
        return True;
"""