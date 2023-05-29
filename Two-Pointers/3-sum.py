class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]

        Logic:
        the for loop runs for every element i, meanwhile the while loop starts with i+1 and last element
        and those two continuously move towards each other, making comparisons in the meantime and
        incrementing left pointer (if i+j+k < 0) or decrementing right pointer (i+j+k > 0) accordingly 
        """
        nums.sort();
        s = set();
        for i in range(len(nums)):
            j = i+1;
            k = len(nums) - 1;
            while j < k:
                sums = nums[i] + nums[j] + nums[k]
                if sums == 0:
                    s.add((nums[i], nums[j], nums[k]));
                    j += 1;
                    k -= 1;
                elif sums < 0:
                    j += 1;
                else:
                    k -= 1;
        return list(s);

solution = Solution();
nums = [-4,-2,-2,-2,0,1,2,2,2,3,3,4,4,6,6];
result = solution.threeSum(nums);
print(result);