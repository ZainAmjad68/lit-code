class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        - Iterate the list and add the number as key and its count as value in a Hash Map
        - Sort the Hash Map by Value
        - Return the lask K keys of the Hash
        """
        resultHash = {};
        for num in nums:
            if num in resultHash:
                resultHash[num] = resultHash[num] + 1;
            else:
                resultHash[num] = 1;

        sortedHash = dict(sorted(resultHash.items(), key=lambda item: item[1]));
        ans = list(sortedHash.keys());
        print(ans);
        return ans[-k:];

solution = Solution();
nums = [1,1,1,2,2,3,3,3,3];
k = 2;

result = solution.topKFrequent(nums,k);
print(result);