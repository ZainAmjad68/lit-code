class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]

        Conditions to Obey:
        # only add open parentheses if open < n
        # only add close parentheses if close < open
        # valid IFF open == close == n
        """
        stack = [];
        result = [];

        def backTrack(openN, closeN):
            if openN == closeN == n:
                result.append("".join(stack));
            
            if openN < n:
                stack.append('(');
                backTrack(openN + 1, closeN);
                stack.pop(); # need to pop because we're using the same list for all the recursive calls

            if closeN < openN:
                stack.append(')');
                backTrack(openN, closeN + 1);
                stack.pop(); # need to pop because we're using the same list for all the recursive calls

        backTrack(0,0);
        return result;


solution = Solution();
n=3;
res = solution.generateParenthesis(n);
print(res);