class Solution:
    def search(self, nums, target):
        l, r = 0, len(nums) - 1

        # for non-existing values, this will be false when at some point, 
        # when either our l will become too big or r become too small
        while l <= r:
            m = l + ((r - l) // 2)  # (l + r) // 2 can lead to overflow
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m
        return -1
        

solution = Solution();
nums = [-1,0,3,5,9,12];
target = 9;
result = solution.search(nums, target);
print(result);