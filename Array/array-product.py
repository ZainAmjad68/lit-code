# prefix = for each element in nums, put the number that we'll get as a result of multiplying the number at that index with all the elements before it.
# e.g.; for a list [1,2,3,4] the prefix at index 3 would be 24 (4 multiplied by 1*2*3)

# postfix = for each element in nums, put the number that we'll get as a result of multiplying the number at that index with all the elements after it.
# e.g.; for a list [1,2,3,4] the prefix at index 1 would be 24 (2 multiplied by 3*4)

class Solution:
    def productExceptSelf(self, nums):
        res = [1] * (len(nums))
        prefix = 1
        # from start to end, calculate the prefixes and save in res
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]
        # res would contain the prefixes here
        postfix = 1
        # from end to start, calculate the postfixes and multiply by prefixes already in res
        for i in range(len(nums) - 1, -1, -1):
            # multiply prefix by postfix to get the answer
            res[i] *= postfix
            postfix *= nums[i]
        return res;

solution = Solution();

nums = [1, 2, 3, 4];
result = solution.productExceptSelf(nums);

# Print the result
print(result);