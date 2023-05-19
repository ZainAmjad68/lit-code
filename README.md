Solving LeetCode problems following the recommended [RoadMap by Neetcode](https://neetcode.io/roadmap).

## Array Problems

### [Contains Duplicate:](https://leetcode.com/problems/contains-duplicate/description/)
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

#### Different Possible Mindsets:
- [Brute Force 'for' loop](https://leetcode.com/problems/contains-duplicate/solutions/2568393/my-1st-attempt-brute-force/)
- [Hash Map](https://leetcode.com/problems/contains-duplicate/solutions/1698064/5-different-approaches-w-explanations/)
- [Sorting the Array](https://leetcode.com/problems/contains-duplicate/solutions/3019357/simple-and-efficient-solution-using-sort-python/).
- [Sets](https://github.com/ZainAmjad68/lit-code/blob/main/Array/contains-duplicate.py)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/contains-duplicate.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (it's slow but concise)
```python
def containsDuplicate(nums):
	return len(set(nums)) != len(nums)
```
### Comments
First thought was to use a brute force for loop that scans the entire array for a duplicate of each element. Then Hash Maps came into mind. And since Sets are just Maps with no repetitions, they were the obvious solution.

**Could've done better. The whole thing took me 20 minutes and the solution was better than 50% in runtime and 10% in space complexity.**

---

### [Valid Anagram:](https://leetcode.com/problems/valid-anagram/description/)
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
#### Different Possible Mindsets:
- [Hash Map](https://github.com/ZainAmjad68/lit-code/blob/main/Array/valid-anagram.py)
- [Sorting both strings and comparing.](https://leetcode.com/problems/valid-anagram/solutions/3132985/python-simple-one-line-solution-explained/)
- [Integer Arrays](https://leetcode.com/problems/valid-anagram/solutions/3261552/easy-solutions-in-java-python-javascript-and-c-look-at-once/)
#### My Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/valid-anagram.py
#### Time/Space Complexity:
- Time complexity: O(2n)
- Space complexity: O(n + n)
### Best Other Solution (it's slow but concise)
```python
def isAnagram(self, s, t):
        return sorted(s.strip()) == sorted(t.strip())
```
### Comments
Started off with Dictionary in mind i.e.; using a loop to add all the letters of 's' in a dictionary and incrementing a count. Then, use another loop to check if the word exists in dictionary, and keep a separate count as well. Return true if all letters exist and count is same.

The above idea evolved to having two dictionaries for both strings, with each containing the count of each letter's frequency in the string so that no separate counters are needed. Then, we simply compare if the two dictionaries are equal.

**One of my better solutions. The complete thing took me 20,25 minutes but the solution was better than 86% in runtime and 97% in space complexity.**

---

### [Two Sum:](https://leetcode.com/problems/two-sum/)
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
#### Different Possible Mindsets:
- [Hash Map](https://github.com/ZainAmjad68/lit-code/blob/main/Array/two-sum.py)
- [Brute Force](https://leetcode.com/problems/two-sum/solutions/3353650/brute-force-solution/)
- [Binary Search](https://leetcode.com/problems/two-sum/solutions/1636227/using-binary-search/)
- [Two Pointer](https://leetcode.com/problems/two-sum/solutions/662/python-dictionary-and-two-pointer-solutions/)
#### My Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/two-sum.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (two pointer; not optimal but easier to visualize)
```python
def twoSum(self, nums, target):
    nums = enumerate(nums)
    nums = sorted(nums, key=lambda x:x[1])
    l, r = 0, len(nums)-1
    while l < r:
        if nums[l][1]+nums[r][1] == target:
            return sorted([nums[l][0], nums[r][0]])
        elif nums[l][1]+nums[r][1] < target:
            l += 1
        else:
            r -= 1
```
### Comments
First thought was to use the Brute Force solution, but we can obviously do better. Then, started thinking about the possibilities of solving it after sorting. First thing that came to mind was Binary Search. But that was an O(n*log(n)) solution at best. Wasn't able to readily implement the Binary Search solution and couldn't think of a better approach so looked at the solution.

**Didn't get the optimized Solution on my own. Mind was stuck on a Binary Search focused solution (which would've taken O(n*logn)). But, after watching the NeetCode solution explanation, i was able to code the O(n) solution in 5,10 minutes and the solution was better than 85% in runtime and 88% in space complexity.**

---

### [Group Anagrams:](https://leetcode.com/problems/group-anagrams/)
Given an array of strings strs, group the anagrams together. You can return the answer in any order.
#### Different Possible Mindsets:
- [Hash Table](https://leetcode.com/problems/group-anagrams/solutions/2384037/python-easily-understood-hash-table-fast-simple/)
- [Single Pass (My Way)](https://github.com/ZainAmjad68/lit-code/blob/main/Array/valid-anagrams.py)
- [Two Solutions using Defaultdict/GroupBy](https://leetcode.com/problems/group-anagrams/solutions/3280005/two-python-solutions-with-result-screenshots/)
#### My Solution (Does Not Pass All Test Cases):
https://github.com/ZainAmjad68/lit-code/blob/main/Array/valid-anagrams.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (easier to visualize but perhaps not the fastest)
```python
def groupAnagrams(self, strs):
    strs_table = {}

    for string in strs:
    	# sort the string
        sorted_string = ''.join(sorted(string))
	
	# if hash key for that string doesn't exist yet, create it and assign it a list
	# that list will contain anagrams belonging to that group
        if sorted_string not in strs_table:
            strs_table[sorted_string] = []
	
	# add the string to the appropriate hash key
        strs_table[sorted_string].append(string)

    return list(strs_table.values())

"""
Visualization:
for an input of type:
strs = ["eat","tea","tan","ate","nat","bat"];
this is how the Hash Table looks:
{'aet': ['eat', 'tea', 'ate'], 'ant': ['tan', 'nat'], 'abt': ['bat']}
"""
```
### Comments
I went with the first approach that came into my mind, which was to take a string and find all of its anagrams (::compare with only those in the list that have the same length::), remove them from the list and form a list of their own and do this again for the remaining strings in the list.

This should be much better than brute force as we skip comparisons b/w strings with unequal length and also remove all the anagrams that belong together in each iteration.

**Seemed like a good solution until some test cases started failing. Nevertheless, came up with the whole approach myself and then coded it within 30 mins so that's decent. But the optimized solution, which uses Sorting and Hashing is much simpler to understand although not better in complexity imo.**

---





