class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # needed this because LeetCode gave the strings in unicode
        # s = s.encode('utf-8');

        # removes non-alphanumeric characters and converts to lowercase
        # filter iterates through each character and check if it satisfies str.isalnum
        # if it does, it is returned and then joined into a new string
        pureS = ''.join(filter(str.isalnum, s)).lower();
        if pureS == "":
            return True;
        index = 0;
        # iterate string from start and end with the same loop
        for left, right in zip(pureS, reversed(pureS)):
            index+=1;
            if left != right:
                return False;
            # if the middle of string is reached, return true
            # cause obv both left and right sides are equal
            if (index == len(pureS)//2 or len(pureS) == 1):
                return True;


solution = Solution();
s = "A man, a plan, a canal: Panama"
result = solution.isPalindrome(s);
print(result);