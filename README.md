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


