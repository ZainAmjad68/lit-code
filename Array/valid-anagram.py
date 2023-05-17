class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        anagram1 = {};
        anagram2 = {};
        for letter in s:
            if letter in anagram1:
                anagram1[letter] = anagram1[letter] + 1;
            else:
                anagram1[letter] = 1;

        for letter in t:
            if letter in anagram2:
                anagram2[letter] = anagram2[letter] + 1;
            else:
                anagram2[letter] = 1;
        
        return anagram1 == anagram2;


solution = Solution();

s = "anagram";
t = "nagaram";

result = solution.isAnagram(s,t);

print(result);