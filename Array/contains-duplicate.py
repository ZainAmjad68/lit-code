class Solution(object):
    def containsDuplicate(self, nums):
        hset = set();
        for num in nums:
            if num in hset:
                return True
            hset.add(num);
        return False;

# Create an instance of the Solution class
solution = Solution();

# Call the containsDuplicate method with a list of numbers
nums = [1, 2, 3];
result = solution.containsDuplicate(nums);

# Print the result
print(result);
