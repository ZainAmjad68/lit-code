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
- [Categorizing Strings by Count (NeetCode)](https://github.com/neetcode-gh/leetcode/blob/main/python/0049-group-anagrams.py)
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

NeetCode solution (given above) is interesting too as it uses a boolean list of all alphabets (with 1's at places where a string character exists and 0's at other places) as the key to hash.
So, for the same input as in best solution above, the hash looks like this:
```
Input: ["eat","tea","tan","ate","nat","bat"]
Resultant Hash Map: 
{
	(1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): ['eat', 'tea', 'ate'], 
	(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): ['tan', 'nat'], 
	(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): ['bat']
}
```
**My approach seemed like a good solution until some test cases started failing. Nevertheless, came up with the whole approach myself and then coded it within 30 mins so that's decent. But the optimized solution, which uses Sorting and Hashing is much simpler to understand although not better in complexity imo.**

---

### [Top K Frequent Elements:](https://leetcode.com/problems/top-k-frequent-elements)
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.
#### Different Possible Mindsets:
- [Hash Map](https://github.com/ZainAmjad68/lit-code/blob/main/Array/top-k-frequent.py)
- [Heap (no sorting so best solution runtime wise)](https://leetcode.com/problems/top-k-frequent-elements/solutions/3246298/347-time-91-58-solution-with-step-by-step-explanation/)
- [Bucket Sort (NeetCode, O(n) solution](https://leetcode.com/problems/top-k-frequent-elements/solutions/3246298/347-time-91-58-solution-with-step-by-step-explanation/)
#### My Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/top-k-frequent.py
#### Time/Space Complexity:
if m is the number of unique elements in the list:
- Time complexity: O(n + m log m + k)
- Space complexity: O(m + k)
### Best Other Solution (uses built in function but very concise)
```python
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    return [x for x,_ in Counter(nums).most_common(k)]
```
### Comments
Went with the first solution that came into my mind as i was sure it would be much better than brute force. It did involve sorting which makes me think there are better solutions out there.

**NeetCode solution is clever but it requires a good amount of thinking (he uses Bucket Sort), my solution was quite intuitive and i was able to code it quite fast too. And its not too bad in terms of complexity.**

---

### [Product of Array except Self:](https://leetcode.com/problems/product-of-array-except-self)
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
#### Different Possible Mindsets:
- [using Division](https://leetcode.com/problems/product-of-array-except-self/solutions/1709002/simple-solution-using-division/)
- [Left Product and Right Product](https://leetcode.com/problems/product-of-array-except-self/solutions/3231758/238-time-96-95-solution-with-step-by-step-explanation/)
- [Prefix and Postfix](https://github.com/ZainAmjad68/lit-code/blob/main/Array/array-product.py)
#### My Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/array-product.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(1)
### Best Other Solution (very efficient but not v intuitive)
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * (len(nums))
        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res
```
### Comments
By far the worst i've done at a problem so far. Except for the simple solution involving Division (which may fail in cases involving 0s), I couldn't think of anything. So, after a bit, I started looking at Solutions and even that took a while to fully sink in. 

**Solution for this is not intuitive, but i've exlpained the thinking behind the solution in my code. Probably will have to keep it somewhere in my mind for future problems. Tricky Question.**

---

### [Valid Sudoku:](https://leetcode.com/problems/valid-sudoku)
Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
- Each row must contain the digits 1-9 without repetition.
- Each column must contain the digits 1-9 without repetition.
- Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
#### Different Possible Mindsets:
- [Naive (Evaluating Rows, Columns and Sub-Boxes separately)](https://leetcode.com/problems/valid-sudoku/solutions/15451/a-readable-python-solution/)
- [Adding all groups in a List and using Set Length](https://leetcode.com/problems/valid-sudoku/solutions/3277043/beats-96-78-short-7-line-python-solution-with-detailed-explanation/)
- [Different Sets for Row, Column and Sub-Box; Each Key in Dict is the col/row no. and Values are the numbers in that col/row (NeetCode)](https://github.com/neetcode-gh/leetcode/blob/main/python/0036-valid-sudoku.py)
#### My Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/valid-sudoku.py
#### Time/Space Complexity:
- Time complexity: O(n^2)
- Space complexity: O(n^2)
### Best Other Solution (intuitive, though takes up a good amount of space)
```python
def isValidSudoku(self, board):
    res = []
    for i in range(9):
        for j in range(9):
            element = board[i][j]
            if element != '.':
                res += [(i, element), (element, j), (i // 3, j // 3, element)]
    # Tuples representing different groups are never equal 
    # (since tuple for row is Tuple[int, str] type, tuple for column is Tuple[str, int] and sub-box - Tuple[int, int, str])
    # So, a Set will eliminate the identical groups so its length will always be less than the length of original list
    return len(res) == len(set(res))
```
### Comments
Have not solved a lot of multi-dimensional problems, nor in practical life or during bachelors. So, couldn't come up with a optimized technique to solve this. Was still able to solve it using the naive approach, evaluating the rows, columns and sub-boxes separately. The solution is slow but it works.

**Multi-Dimensional Problems can be confusing. Solutions given were similar in nature, using Sets to evaluate if there's any repitition. 
My solution was better than only 6% at runtime and 50% at space complexity. So, should definitely use optimized techniques to solve in future.**

---

### [Encode and Decode Strings:](https://www.lintcode.com/problem/659)
Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.
#### Different Possible Mindsets:
- [My Approach](https://github.com/ZainAmjad68/lit-code/blob/main/Array/encode-decode.py)
- [NeetCode (similar approach)](https://github.com/neetcode-gh/leetcode/blob/main/python/0271-encode-and-decode-strings.py)
- [A Medium Solution](https://medium.com/@miniChang8/leetcode-encode-and-decode-strings-4dde7e0efa1c)
#### My Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/encode-decode.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (similar to mine, may have a bit less space complexity)
```python
def encode(self, strs):
    res = ""
    for s in strs:
        res += str(len(s)) + "#" + s
    return res

def decode(self, s):
    res, i = [], 0

    while i < len(s):
        j = i
        while s[j] != "#":
            j += 1
        length = int(s[i:j])
        res.append(s[j + 1 : j + 1 + length])
        i = j + 1 + length
    return res
```
### Comments
Didn't understand the problem at first as it's an odd one compared to others. There's really only one approach that makes sense and most solutions on internet follow that approach. (encoding/decoding in format lengthOfWord + a delimiter + the actual word i.e.; ['i', 'am', 'groot'] converts to 1#i2#am5#groot and vice versa.)
**Once I understood the problem and the approach required, it took 15 mins to code the solution. And it's similar to other solutions on the internet.**

---

### [Longest Consecutive Sequence:](https://leetcode.com/problems/longest-consecutive-sequence)
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.
#### Different Possible Mindsets:
- [Brute Force](https://leetcode.com/problems/longest-consecutive-sequence/solutions/2238932/c-python-simple-solution-w-explanation-o-n-o-n/)
- [Sorting](https://github.com/ZainAmjad68/lit-code/blob/main/Array/longest-consecutive-sequence.py)
- [Set](https://github.com/neetcode-gh/leetcode/blob/main/python/0128-longest-consecutive-sequence.py)
#### My Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Array/longest-consecutive-sequence.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (easy to understand, and best runtime as well)
```python
def longestConsecutive(self, nums):
    numSet = set(nums)
    longest = 0

    for n in nums:
        if (n - 1) not in numSet:
            length = 1
            while (n + length) in numSet:
                length += 1
            longest = max(length, longest)
    return longest
```
### Comments
Could Not figure out a O(n) solution, so decided to go ahead with the next best one that i knew i.e.; Sorting and then a bunch of if statements. Coded that in 10 mins, but handling edge cases afterwards took ~30 mins. :(


**The Optimal Solution was so simple but not intuitive (to me). Hopefully, i'll start to make these connections in the future once i've solved enough problems.
Still, my current solution was better than 52% in runtime and 89% in space complexity so i did okay.**

---




## Two Pointer Problems

### [Valid Palindrome:](https://leetcode.com/problems/valid-palindrome)
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.


Given a string s, return true if it is a palindrome, or false otherwise.

#### Different Possible Mindsets:
- [Reverse the String](https://leetcode.com/problems/valid-palindrome/solutions/3024037/python-regex-explained-beats-99/)
- [Two Pointer (start -> <- end)](https://leetcode.com/problems/valid-palindrome/solutions/3524673/c-java-python-javascript-simple-code-easy-to-understand/)
- [NeetCode (has its own non-alphanum function)](https://github.com/neetcode-gh/leetcode/blob/main/python/0125-valid-palindrome.py).
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Two%20Pointers/valid-palindrome.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(1)
### Best Other Solution (concise, uses regex, reverses and equates)
```python
def isPalindrome(self, s):
        new_s = re.sub(r"[^a-zA-Z0-9\\s+]", "", s).lower();
        return new_s == new_s[::-1];
```
### Comments
Recognized that Two Pointer solution would be better. Wasted a lot of time looking for a way to iterate string from start and end with the same loop, when i could've used a `while left < right` loop.

Learned some stuff about Regex and Zip which is good.

**Had the right approach. Googling wasted some time when i could've used a simple while loop. Good effort still, but NeetCode solution is more interview friendly though. But to be fair, my solution was better than 89% in runtime but just 5% in space.**

---


### [Two Sum II - Input Array Is Sorted:](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/)
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 < numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

#### Different Possible Mindsets:
- [Brute Force](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/solutions/1756172/python-from-brute-force-optimized-solution-with-intuitive-explaination/)
- [Binary Search](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/solutions/2682118/python-binary-search-o-n-log-n-time-o-1-space-clean-code/)
- [Dictionary](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/solutions/51249/python-different-solutions-two-pointer-dictionary-binary-search/).
- [Two Pointers](https://github.com/neetcode-gh/leetcode/blob/main/python/0167-two-sum-ii-input-array-is-sorted.py)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Two-Pointers/two-sum-pt2-sorted-input-array.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(1)
### Best Other Solution (concise, uses regex, reverses and equates)
```python
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    l, r = 0, len(numbers) - 1

    while l < r:
        curSum = numbers[l] + numbers[r]

        if curSum > target:
            r -= 1
        elif curSum < target:
            l += 1
        else:
            return [l + 1, r + 1]
```
### Comments
Solution was rather simple but i made it complicated in my head. First, i thought about moving two pointers along with each other. Then, having one pointer at start and other one at end made more sense. And that's how NeetCode had done it as well.

**Had the right approach. Complicated it for myself which wasted time. Solution was better than 55% in runtime and 89% in space complexity.**

---


### [Three Sum:](https://leetcode.com/problems/3sum)
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

#### Different Possible Mindsets:
- [Two Pointer](https://leetcode.com/problems/3sum/solutions/3109452/c-easiest-beginner-friendly-sol-set-two-pointer-approach-o-n-2-logn-time-and-o-n-space/)
- [Handle different cases in Individual Steps](https://leetcode.com/problems/3sum/solutions/725950/python-5-easy-steps-beats-97-4-annotated/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Two-Pointers/3-sum.py
#### Time/Space Complexity:
- Time complexity: O(n^2 logn)
- Space complexity: O(n)
### Best Other Solution (NeetCode one)
```python
def ThreeSum(self, integers):
    """
    :type integers: List[int]
    :rtype: List[List[int]]
    """
    integers.sort()
    result = []
    for index in range(len(integers)):
        if integers[index] > 0:
            break
        if index > 0 and integers[index] == integers[index - 1]:
            continue
        left, right = index + 1, len(integers) - 1
        while left < right:
            if integers[left] + integers[right] < 0 - integers[index]:
                left += 1
            elif integers[left] + integers[right] > 0 - integers[index]:
                right -= 1
            else:
                result.append([integers[index], integers[left], integers[right]]) # After a triplet is appended, we try our best to incease the numeric value of its first element or that of its second.
                left += 1 # The other pairs and the one we were just looking at are either duplicates or smaller than the target.
                right -= 1 # The other pairs are either duplicates or greater than the target.
                # We must move on if there is less than or equal to one integer in between the two integers.
                while integers[left] == integers[left - 1] and left < right:
                    left += 1 # The pairs are either duplicates or smaller than the target.
    return result
```
### Comments
I was close to the solution, but was struggling with visualizing a while loop inside the for loop. Looking at a solution made things clear. Questions involving triplets can be a bit confusing, need some focus.

**Pretty much the same single approach is being used in all of the solutions i.e.; the for loop runs for every element i, meanwhile the while loop starts with i+1 and last element and those two continuously move towards each other.**

---


### [Container With Most Water:](https://leetcode.com/problems/container-with-most-water/description/)
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

#### Different Possible Mindsets:
- [Brute Force](https://leetcode.com/problems/container-with-most-water/solutions/1915744/python-brute-force-two-pointer/)
- [Two Pointer](https://github.com/neetcode-gh/leetcode/blob/main/python/0011-container-with-most-water.py)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Two-Pointers/container-with-most-water.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (not as readable, but concise)
```python
def maxArea(self, H: List[int]) -> int:
    ans, i, j = 0, 0, len(H)-1
    while (i < j):
        if H[i] <= H[j]:
            res = H[i] * (j - i)
            i += 1
        else:
            res = H[j] * (j - i)
            j -= 1
        if res > ans: ans = res
    return ans
```
### Comments
LeetCode explanation of the problems sucked, so i looked at the discussion to understand the problem and once i did, the solution came fairly easy. Coded the optimal solution in 6,7 mins.

**Got the Solution easily prolly because my head was clear which goes a long way. The solution was better than 52% in runtime and 89% in space.**

---



### [Trapping Rain Water:](https://leetcode.com/problems/trapping-rain-water/description/)
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

#### Different Possible Mindsets:
- [Brute Force O(n) space and time](https://leetcode.com/problems/trapping-rain-water/solutions/3387829/c-java-python-javascript-o-n-time-o-1-space-brute-force-optimized-code/)
- [Dynamic Programming](https://leetcode.com/problems/trapping-rain-water/solutions/2589802/leetcode-the-hard-way-explained-line-by-line/)
- [Two Pointer (Optimized O(1) space)](https://github.com/neetcode-gh/leetcode/blob/main/python/0042-trapping-rain-water.py)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Two-Pointers/trapping-rain-water.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(1)
### Best Other Solution (not as readable, but concise)
```python
def trap(self, height: List[int]) -> int:
    i, j, ans, mx, mi = 0, len(height) - 1, 0, 0, 0
    # two pointers 
    # pointer i from the left
    # pointer j from the right
    while i <= j:
        # take the min height
        mi = min(height[i], height[j])
        # find the max min height
        mx = max(mx, mi)
        # the units of water being tapped is the diffence between max height and min height
        ans += mx - mi
        # move the pointer i if height[i] is smaller
        if height[i] < height[j]: i += 1
        # else move pointer j
        else: j -= 1
    return ans
```
### Comments
Not easy to come up with the Algo on your own, but quite easy to understand the Brute Force Approach [O(n) space]. The optimized approach is a bit more tricky but still understandable.

**Was able to code once I understood the algo, Will need to keep it in mind. Solution was better than 70% in runtime and 90% in space complexity.**

---





## Stack Problems

### [Valid Parentheses:](https://leetcode.com/problems/valid-palindrome)
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
An input string is valid if:
* Open brackets must be closed by the same type of brackets.
* Open brackets must be closed in the correct order.
* Every close bracket has a corresponding open bracket of the same type.

#### Different Possible Mindsets:
- [Using Stack and HashMap](https://github.com/neetcode-gh/leetcode/blob/main/python/0020-valid-parentheses.py)
- [Using Python Replace](https://leetcode.com/problems/valid-parentheses/solutions/885074/python-solution-in-5-lines/)
- [Using Stack Only)](https://leetcode.com/problems/valid-parentheses/solutions/3399077/easy-solutions-in-java-python-and-c-look-at-once-with-exaplanation/).
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/valid-parentheses.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (concise and easy to understand)
```python
def isValid(self, s):
    """
    :type s: str
    :rtype: bool
    """
    d = {'(':')', '{':'}','[':']'}
    stack = []
    for i in s:
        if i in d:  # 1
            stack.append(i)
        elif len(stack) == 0 or d[stack.pop()] != i:  # 2
            return False
    return len(stack) == 0 # 3

# 1. if it's the left bracket then we append it to the stack
# 2. else if it's the right bracket and the stack is empty(meaning no matching left bracket), or the left bracket doesn't match
# 3. finally check if the stack still contains unmatched left bracket
```
### Comments
Stack based solution didn't come into mind so i went with using a Dictionary to keep count of each symbol and then returning true if the number of opening symbols of a specific parentheses are equal to its closing symbols. This sounded good on paper but was difficult to implement. Then, took a look at Stack solution and it was quite intuitive and easy so went with that.

**Dictionary based attempt took way too long, a lot of edge cases. Should've pivoted to Stack based solution much earlier as it was easier and much more concise. Stack solution surpassed 85% in runtime and 90% in space.**


---

### [Min Stack:](https://leetcode.com/problems/min-stack/)
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
**- int getMin() retrieves the minimum element in the stack.**
You must implement a solution with **O(1) time complexity** for each function.

#### Different Possible Mindsets:
- [Using Separate Stack for Min](https://github.com/neetcode-gh/leetcode/blob/main/python/0155-min-stack.py)
- [Use a Single Stack for both Val and Min (Hint: Make use of Tuples)](https://leetcode.com/problems/min-stack/solutions/3545225/c-java-python-javascript-simple-solution-with-o-1-time-complexity-for-each-function-as-asked/)
- [Using a separate Deque)](https://leetcode.com/problems/min-stack/solutions/2812426/python-solution-with-explanation/).
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/min-stack.py
#### Time/Space Complexity:
- Time complexity: O(1)
- Space complexity: O(n)
### Best Other Solution (faster and less space complexity because of single stack)
```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val):
        if not self.stack:
            self.stack.append((val, val))
        else:
            mn = min(self.stack[-1][1], val)
            self.stack.append((val, mn))

    def pop(self):
        if self.stack:
            self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1][0]
        return 0

    def getMin(self):
        if self.stack:
            return self.stack[-1][1]
        return 0
```
### Comments
First thought was to use a min variable, and update the min value on push. But then realized that when we pop something, there's a chance that min value is no longer accurate so we have to update it there too. But, how can we do it in O(1)? We can't. Will have to go through the stack to find the new min.

Then, saw the comments talking about keeping a separate min stack and manipulating it on push and pop. So, that is the approach i used.

**Intuitive to go for a separate min value to keep track. But once you realize the problems with that, a separate stack is the obvious solution. There's another cheeky way to [use tuples with a single stack](https://leetcode.com/problems/min-stack/solutions/3545225/c-java-python-javascript-simple-solution-with-o-1-time-complexity-for-each-function-as-asked/) to achieve the same thing. My solution with two stacks was better than 65% at runtime and 55% at space.**

---


### [Evaluate Reverse Polish Notation:](https://leetcode.com/problems/evaluate-reverse-polish-notation)
You are given an array of strings tokens that represents an arithmetic expression in a [Reverse Polish Notation](http://en.wikipedia.org/wiki/Reverse_Polish_notation).

Evaluate the expression. Return an integer that represents the value of the expression.

#### Different Possible Mindsets:
- [Stack (push current answer in there as well](https://leetcode.com/problems/evaluate-reverse-polish-notation/solutions/2920598/easy-solution-w-explanation-c-java-python-no-runtime-error/)
- [Recursive Approach (not intuitive at all, but neat)](https://leetcode.com/problems/evaluate-reverse-polish-notation/solutions/2920607/python-easy-solution-faster-than-80/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/reverse-polish-notation.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (faster and less space complexity, neat as well)
```python
class Solution(object):
    def evalRPN(self, tokens):
        s=[]                     #assign a empty stack s
        for i in tokens:
            if i not in '-+/*':
                s+=[int(i)]              #you can also use s.append()
            else:
                a=int(s.pop())         
                b=int(s.pop())
                if i=='+':
                    s+=[b+a]
                elif i=='*':
                    s+=[b*a]
                elif i=='-':
                    s+=[(b-a)]
                else:
                    s+=[b/a]
        return int(s[-1])          #data type int is used for float value
```
### Comments
My Approach was quite close to the one being used in most Solutions, one clever thing they did was to push the result back into the stack, which i didn't do and thus had to handle a bunct of edge cases (pop 2 from stack first time, and then 1 afterwards etc.) separately.

The division operation caused some issues as its done differently on different languages (or even different versions of the same language!).

**A good stack problem, and i was close to optimal solution, so a good effort. In the future, should think of ways to optimize the general case (push current result in the stack in this case) to handle edge cases as well instead of handling them separately.**

---


### [Generate Parentheses:](https://leetcode.com/problems/generate-parentheses/description/)
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
Example:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

#### Different Possible Mindsets:
- [DFS (Recursive Approach)](https://leetcode.com/problems/generate-parentheses/solutions/2542620/python-java-w-explanation-faster-than-96-w-proof-easy-to-understand/)
- [Backtracking](https://github.com/neetcode-gh/leetcode/blob/main/python/0022-generate-parentheses.py)
- [Using Stack](https://leetcode.com/problems/generate-parentheses/solutions/2712761/beautiful-iterative-python-solution-with-stack/)
- [Some Random Algos that work](https://leetcode.com/problems/generate-parentheses/solutions/3260192/three-python-solution-that-beats-100-and-91-90-and-93-3/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/generate-parentheses.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (uses same intuition as NeetCode, but also utilizes the fact that each combination will be of length n*2)
```python
def generateParenthesis(self, n):
    """
    :type n: int
    :rtype: List[str]
    """

    def dfs(left,right,s):
        print(s);
        if len(s)==n*2:
            return result.append(s) 
        
        if left<n:
            dfs(left+1,right,s+'(')
        
        if right< left:
            dfs(left,right+1,s+')')
    
    result=[]
    dfs(0,0,'')
    return result
```
### Comments
Only solution I could come up with was to create a list with all combinations of n paranthesis and eliminate the invalid ones, but this would essentially be n^n solution, so very bad.

NeetCode used Backtracking, and by obeying just some simple intuitive rules (Add `(` if `( < n`, Add `)` if `) < (` and return IFF `( == ) == n`), the solution became quite simple and efficient as well.

**Somewhat Familiar with Backtracking, but not enough practical exposure for the mind to go there when i saw this problem. Need to learn more about that and figure out the kind of problems where it can be used.**

---


### [Daily Temperatures:](https://leetcode.com/problems/daily-temperatures/description/)
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

Example:
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

#### Different Possible Mindsets:
- [Brute Force](https://github.com/ZainAmjad68/lit-code/blob/main/Stack/min-stack.py)
- [Using Stack (containing [Temp, Index])](https://github.com/neetcode-gh/leetcode/blob/main/python/0739-daily-temperatures.py)
- [Using Stack (containing only Index)](https://leetcode.com/problems/daily-temperatures/solutions/2506436/python-stack-97-04-faster-simplest-solution-with-explanation-beg-to-adv-monotonic-stack/)
- [Without Stack (O(n) solution)](https://leetcode.com/problems/daily-temperatures/solutions/838903/python-o-n-time-and-o-1-space-without-stack/)
- [Many Different Approaches](https://leetcode.com/problems/daily-temperatures/solutions/275884/python-4-different-solutions-from-easy-to-elegant/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/min-stack.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (both are fast, but different approaches)
```python
# most concise version of the common solution, using a Stack
class Solution:
    def dailyTemperatures(self, T):
        ans = [0] * len(T)
        stack = []
        for i, t in enumerate(T):
          while stack and T[stack[-1]] < t:
            cur = stack.pop()
            ans[cur] = i - cur
          stack.append(i)
        return ans
```

```python
# fastest and without using a Stack
class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        ans = [0]*len(temperatures)
        hottest = 0
        for i in range(len(temperatures)-1,-1,-1):
            temp = temperatures[i]
            if temp >= hottest:
                hottest = temp
                continue
            days = 1
            while temperatures[i+days]<=temp:
                days+=ans[i+days]
            ans[i] = days
        return ans
```

### Comments
Came Up quite easily with the naive approach i.e.; a nested loop, where for each element, we find the index that has a value greater than that element and see how far away that value is (index where such value is found - current index). However, this solution exceeded the time limit for bigger inputs so not good enough.

NeedCode's approach is O(n) and included a stack of type [temp, index] where each value is popped if it is smaller than the current Temp.

**Need to consider using Tuples to store multiple pieces of info at one point/index of Stack OR Hash. Couldn't figure out a soltuion other than Naive because of that, cause i couldn't figure out how to keep track of index in stack. But devised the Naive solution relatively fast so decent job.**

---



### [Car Fleet:](https://leetcode.com/problems/car-fleet/description/)
There are n cars going to the same destination along a one-lane road. The destination is target miles away.

You are given two integer array position and speed, both of length n, where position[i] is the position of the ith car and speed[i] is the speed of the ith car (in miles per hour).

A car can never pass another car ahead of it, but it can catch up to it and drive bumper to bumper at the same speed. The faster car will slow down to match the slower car's speed. The distance between these two cars is ignored (i.e., they are assumed to have the same position).

A car fleet is some non-empty set of cars driving at the same position and same speed. Note that a single car is also a car fleet.

If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.

Return the number of car fleets that will arrive at the destination.

#### Different Possible Mindsets:
- [Increasing Stack](https://leetcode.com/problems/car-fleet/solutions/1537985/python3-increasing-stack/)
- [Without Stack (see comment)](https://leetcode.com/problems/car-fleet/solutions/255589/python-code-with-explanations-and-visualization-beats-95/)
- [Greedy Pattern](https://leetcode.com/problems/car-fleet/solutions/1299193/python-greedy-pattern-explained/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/car-fleet.py
#### Time/Space Complexity:
- Time complexity: O(nlog(n))
- Space complexity: O(n)
### Best Other Solution (simple and fast)
```python
# fastest and without using a Stack
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        prev_t = None
        n = 0
        for pos, vel in sorted(zip(position, speed))[::-1]:
            dist = target - pos
            t = dist / vel
            if not prev_t or t > prev_t:
                prev_t = t
                n += 1
        return n
```

### Comments
My intuition was that if a car meets with another car, they move at the speed of the slower car among them. So, moving at that speed, if another car cathes up to them, the above process repeats. And so the loop runs until all the cars have reached the target
For each iteration, each car takes the next step i.e.; move position by its speed, and then we check if any of the cars coincide. if they do, we update the speed by min(car x, car y, car ..);
We also check if any fleet has reached the destination and if so, we increase a counter We have to count how many times car/s reach the destination (whether together in a fleet or by themselves).

However, the above ignores the fact that we just need to check which cars are together at the target, so we can just compare the times of different cars reaching the target.


NeedCode's approach is to take this as a system of linear equations, with time on X-Axis and position on Y-Axix, whereas the speed is the slope of how it moves. And that slope will tell us whether any cars meet at some point.

The actual solution is to sort the cars by position and start by the furthest car. Then, check if the car behind it will reach the destination before this car. If yes, they're going to collide at some point and become a fleet. So, remove the car that's behind since it will slow down to the speed of the car ahead when they collide and continue the comparion of the furthest car with the next car in behind.
And using the same logic, if the behind car takes more time than the head of the fleet => that means that it will not be fleet with the head car => meaning we have new fleet started.

****Use (Target - Position)/Speed to get the time a car will reach its destination****

**I was making the problem more complicated, by checking everything at each position. But if we just solve for the target, things become much more simple.**

---


### [Largest Rectangle in Histogram:](https://leetcode.com/problems/largest-rectangle-in-histogram)
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

#### Different Possible Mindsets:
- [Stack](https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/28917/ac-python-clean-solution-using-stack-76ms/)
- [Monotone Increasing Stack](https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/688492/python-monotone-increasing-stack-similar-problems-attached/)
- [Divide and Conquer](https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/894294/python-divide-conquer-with-comments-o-nlogn/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/largest-rectangle-histogram.py
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (simple and fast)
```python
def largestRectangleArea(self, height):
    height.append(0)
    stack = [-1]
    ans = 0
    for i in xrange(len(height)):
        while height[i] < height[stack[-1]]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1
            ans = max(ans, h * w)
        stack.append(i)
    height.pop()
    return ans
```

### Comments

Very Hard to wrap your head around. The main intuition is to that once we know that we can't extend the area of a bar any further (i.e.; bars are in decreasing trajectory towards the right), we pop it and see if its area was higher than the max one so far, if so, assign it to max. And since we only pop the most recent bars and then go from there, a stack is a good Data Structure to use.

Here are some explanations:
- https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/28917/ac-python-clean-solution-using-stack-76ms/comments/492440
- https://leetcode.com/problems/largest-rectangle-in-histogram/solutions/28917/ac-python-clean-solution-using-stack-76ms/comments/186913

**Watched the LeetCode solution and solved it, but unlikely that i'll be able to retain it in memory.**

---

### [Reverse Substrings Between Each Pair of Parentheses:](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/)
You are given a string s that consists of lower case English letters and brackets.
Reverse the strings in each pair of matching parentheses, starting from the innermost one.
Your result should not contain any brackets.

#### Different Possible Mindsets:
- [Stack](https://github.com/ZainAmjad68/lit-code/blob/main/Stack/reverse-substrings-between-parentheses.js)
- [Regex](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/solutions/818312/javascript-beats-98-of-the-runtime-with-comments/)
- [Recursion](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/solutions/2290806/python-sol-recursion-and-stack-sol-detailed-explanation-with-pictrue/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Stack/reverse-substrings-between-parentheses.js
#### Time/Space Complexity:
- Time complexity: O(n)
- Space complexity: O(n)
### Best Other Solution (simple, but uses Regex)
```js
function reverseInParentheses(inputString) {
    while (inputString.includes("(")) {
        inputString = inputString.replace(/\(([^()]*)\)/, (_, contents) => {
            return [...contents].reverse().join('');
        });
    }
    return inputString;
}
```

### Comments

Once you remember to use Stack for Nested Parentheses, that solution is pretty intuitive and simple. Couldn't come up with this on my own as i was making it too complicated, but wrote the solution by hand once i understood it.

**Stack solution is easy and does not contain any RegEx, ideal for interview situations.**

---

## Binary Search Problems

### [Binary Search:](https://leetcode.com/problems/binary-search)
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

#### Different Possible Mindsets:
- [Two Pointers and While Loop](https://leetcode.com/problems/binary-search/solutions/423162/binary-search-101/)
- [Recursion](https://leetcode.com/problems/binary-search/solutions/2573688/python-binary-search/)
- [Array Slicing (Bad)](https://leetcode.com/problems/binary-search/solutions/3131273/worst-log-n-recursive-solution-python/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/binary-search.py
#### Time/Space Complexity:
- Time complexity: O(logn)
- Space complexity: O(n)
### Best Other Solution (difficult to understand, but a two liner)
```python
def search(self, nums, target):
    index = bisect.bisect_left(nums, target)
    return index if index < len(nums) and nums[index] == target else -1
```
### Comments
Tried to do through finding Middle and then Array Slicing, but ran into problems. Then, used the two pointer method, which is quite simple and also easy to implement.

**Didn't have the Two Pointer solution in mind, also was trying for recursion but figured out after a bit that a recursive solution is easier with a helper function (difficult to track index otherwise).**

---

### [Search 2D Matrix:](https://leetcode.com/problems/search-a-2d-matrix)
You are given an m x n integer matrix matrix with the following two properties:

Each row is sorted in non-decreasing order.
The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.

#### Different Possible Mindsets:
- [Traditional Binary Search Modified](https://leetcode.com/problems/search-a-2d-matrix/solutions/26201/a-python-binary-search-solution-o-logn/)
- [Some Short but not very intuitive solutions](https://leetcode.com/problems/search-a-2d-matrix/solutions/1895915/python-3-short-and-readable-solutions/)
- [My Way (not a common algorithm)](https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/search-2d-matrix.py)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/search-2d-matrix.py
#### Time/Space Complexity:
- Time complexity: O(log (n*m))
- Space complexity: O(n)
### Best Other Solution (O(m+n), just checks each rows for the existence of target)
```python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    for row in matrix:
        if row[-1] >= target:
            return target in row
    return False
```
### Comments
A good problem to solve after you've just refreshed Binary Search in your mind, you get to apply it once you find where you need to do so.

Algorithm is quite simple:
- First Check if the target even exists in the matrix, by checking start and end of the matrix (as its sorted in increasing order)
- Then, if target is within bounds, find the row where the target might exist
- Run Binary Search on that Row

**Proud that i came up with a solution all by myself that got accepted on the first try. I took 5,6 mins to think about how we could solve this, it was clear that we needed to run binary search on a row. I just needed to come up with a way to find that row.

The solution was 64% faster in runtime than other submissions, and took less space than 23% of them.**

---

### [Koko Eating Bananas:](https://leetcode.com/problems/koko-eating-bananas/)
Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

#### Different Possible Mindsets:
- [Brute Force](https://leetcode.com/problems/koko-eating-bananas/solutions/1705145/python-binarysearch-optimizations-explained/)
- [Binary Search + Optimizations](https://leetcode.com/problems/koko-eating-bananas/solutions/1705145/python-binarysearch-optimizations-explained/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/koko-eating-bananas.py
#### Time/Space Complexity:
- Time complexity: O(log(max(n))*n)
- Space complexity: O(n)
### Best Other Solution (same approach, more concise)
```python
def minEatingSpeed(self, piles, H):
    l, r = 1, max(piles)
    while l < r:
        m = (l + r) / 2
        if sum((p + m - 1) / m for p in piles) > H:
            l = m + 1
        else:
            r = m
    return l
```
### Comments
Couldn't think of a way except brute force, but watching NeetCode made me realize how it's possible to derive a Binary Search solution from Naive one.


Approach:	
Brute Force is basically checking all values from 1 to max(pile) to see if it can finish the piles in h time, so O(max(pile)*pile) because in worst case we'll find that all piles can only be eaten if koko eats one pile each hour [and to eat the biggest pile in an hour, it'll need a speed of max(pile)]
        

But instead of going through every value from 1 to max(pile), we use Binary Search to narrow down and reduce computation O(log(max(pile))*pile).

However, this solution, though working still exceeded time limit so perhaps some other approaches might be better.

---

### [Find Minimum in Rotated Sorted Array:](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/)
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

#### Different Possible Mindsets:
- [Intuitive](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/solutions/1436502/python-binary-search-with-picture-clean-concise/)
- [Most Common](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/solutions/396767/simple-binary-search/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/min-in-rotated-sorted-array.py
#### Time/Space Complexity:
- Time complexity: O(log(n))
- Space complexity: O(n)
### Best Other Solution (same approach, more concise)
```python
def findMin(self, nums): # if no time constraint
	return min(nums)
```
### Comments

The Solution seen commonly is concise but might not come naturally. Even after you've seen it, it takes a couple of visual runs of the code to properly understand it.

**Need to keep in mind the trick/technique for these type of questions.**

---

### [Search In Rotated Sorted Array:](https://leetcode.com/problems/search-in-rotated-sorted-array/)
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

#### Different Possible Mindsets:
- [Traversal](https://leetcode.com/problems/search-in-rotated-sorted-array/solutions/2831184/beats-99-submissions-fastest-and-easiest-one-liner/)
- [Binary Search](https://leetcode.com/problems/search-in-rotated-sorted-array/solutions/2759124/most-efficient-solution-binary-search-modified-for-rotated-sorted-list/)
- [Using Bisect Module](https://leetcode.com/problems/search-in-rotated-sorted-array/solutions/1732232/python-using-bisect-with-a-key-function/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/search-in-rotated-sorted-array.py
#### Time/Space Complexity:
- Time complexity: O(log(n))
- Space complexity: O(n)
### Best Other Solution (concise and relatively easy)
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if target in nums :
            return nums.index(target)
        else :
            return -1
```
### Comments

Works by repeatedly checking which portion of the array (left or right) is sorted and then adjusting the pointers to continue the search in the appropriate direction.

**A not so straight forward problem. Not many approaches besides Binary Search to solve it either.**

---

### [Time Based Key Value Store:](https://leetcode.com/problems/time-based-key-value-store/description/)
Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

TimeMap() Initializes the object of the data structure.
void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".

#### Different Possible Mindsets:
- [Linear Search](https://leetcode.com/problems/time-based-key-value-store/solutions/2673112/python-dictionary-linear-search/)
- [Binary Search](https://leetcode.com/problems/time-based-key-value-store/solutions/408651/python-clean-solution-binary-search/)
- [Using Bisect Module](https://leetcode.com/problems/time-based-key-value-store/solutions/1223226/python3-clean-and-efficient-code-o-1-o-log-n/)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/time-based-key-value-store.py
#### Time/Space Complexity:
- Time complexity: O(log(n))
- Space complexity: O(n)
### Best Other Solution (concise but not intuitive at all)
```python
class TimeMap:
    def __init__(self):
        self.times = collections.defaultdict(list)
        self.values = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.times[key].append(timestamp)
        self.values[key].append(value)

    def get(self, key: str, timestamp: int) -> str:
        i = bisect.bisect(self.times[key], timestamp)
        return self.values[key][i - 1] if i else ''
```
### Comments

A simple problem but the description is not well stated. Was able to code in 10,15 mins. The strictly increasing timestamp constraint makes sense (time is always increasing so no need to sort for Binary Search) but is a bit confusing because we're using _int_ as timestamp here (which isn't guaranteed to increase strictly).


**One of the few problems where Binary Search just makes sense. Though my solution, even with binary search, was only faster than 15% and took less space than 50%.**

---

### [Median of Two Sorted Arrays:](https://leetcode.com/problems/median-of-two-sorted-arrays/)
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

#### Different Possible Mindsets:
- [Merge Sort (not optimal but easier to understand)](https://leetcode.com/problems/median-of-two-sorted-arrays/solutions/3431113/merge-sort-approach/)
- [Binary Search](https://leetcode.com/problems/median-of-two-sorted-arrays/solutions/3489494/c-java-python-javascript-o-log-min-m-n-iterative-solution-with-explanation/)
- [Binary Search Alternate (easier to digest)](https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/median-of-sorted-arrays.py)
#### Solution:
https://github.com/ZainAmjad68/lit-code/blob/main/Binary-Search/median-of-sorted-arrays.py
#### Time/Space Complexity:
- Time complexity: O(log(m+n))
- Space complexity: O(m+n)
### Best Other Solution (not O(log (m+n) but concise)
```python
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    merge=sorted(nums1+nums2);
    l=len(merge);
    if l%2==0:
        return (merge[l//2]+merge[l//2-1])/2
    return merge[l//2]
```
### Comments

FYI, median is the middle value, and if there are two middle values, then its their average.
Simplest Solution is to merge the arrays and then compute but merging take O(m+n) time so that's out of the window.


**Optimal Solution Intuition:**
Given two sorted arrays nums1 and nums2 of size n1 and n2 respectively, the median of the combined sorted array can be found by dividing the elements of the combined sorted array into two halves of equal length such that one half is always greater than the other. The median is then the average of the maximum element of the left half and the minimum element of the right half if the total number of elements is even, or the maximum element of the left half if the total number of elements is odd.


**Algorithm (Hint: We run Binary Search only on the smaller array):**
Find middle of the smaller array, and consider elements on the left as the left partition. Now, find left partition of bigger array by using: _(total elements in both arrays - size of left partition in smaller array)_.
To check if our partitions are correct, check if first element in the right partition of smaller array is smaller or equal to first right partition element in bigger array and vice versa. If yes, we're good otherwise, make the left partition of the smaller array one element smaller if its right most element was bigger or make the left partition element bigger if its right most element was smaller and check again.
Now we get the right partition, find the median by taking: _(min(last element in left partition of both arrays) + max(first element in right partition of both arrays))/2_

**Hard to come up with this solution on the spot, but if solved before, it is intuitive enough to make decent progress during an interview.**

---

