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

    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        finalList = [];
        for st in strs:
            sameLength = [s for s in strs if (len(s) == len(st) and s != st)];

            # if no other element in list is of same length then there's no anagram
            if not sameLength:
                finalList.append([st]);
                continue;

            localList = [st];
            # if there are anagrams, add them to the final list
            # and remove them from input list so that we don't iterate on them again
            for sl in sameLength:
                isAna = self.isAnagram(st, sl);
                print(isAna);
                if isAna:
                    localList.append(sl);
                    strs.remove(sl);
                print(localList);
            finalList.append(localList);
        return finalList;

solution = Solution();
strs = ["eat","tea","tan","ate","nat","bat"];
result = solution.groupAnagrams(strs);
print(result);
