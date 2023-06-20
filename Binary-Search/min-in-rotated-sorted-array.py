class Solution:
    def findMin(self, nums):
        first, last = 0, len(nums) - 1
        '''        
        loop invariant: 1. low < high
                        2. mid != high and thus A[mid] != A[high] (no duplicate exists)
                        3. minimum is between [low, high]
        The proof that the loop will exit: 
        after each iteration either the 'high' decreases or the 'low' increases, so the interval [low, high] will always shrink.
        '''
        while first < last:
            midpoint = (first + last) // 2
            print(first);
            print(last);
            print(midpoint);

            if nums[midpoint] > nums[last]:
                # the minimum is in the right part
                first = midpoint + 1
            else:
                # the minimum is in the left part
                last = midpoint
        return nums[first]
    
solution = Solution();
nums = [3,5,9,12,0,1];
result = solution.findMin(nums);
print(result);