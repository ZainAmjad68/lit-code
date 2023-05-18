class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashMap = {};
        for index, num in enumerate(nums):
            difference = target - num;
            if difference in hashMap:
                return [hashMap[difference], index];
            else:
                hashMap[num] = index;

solution = Solution();
nums = [3,2,5,7,4];
target = 6;
result = solution.twoSum(nums, target);
print(result);