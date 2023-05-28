class Solution(object):
    def twoSum(self, numbers, target):
        """
        begin from start and end of array
        if start+end too big, we need to look at smaller numbers
        since array is sorted, numbers towards the end are biggest
        so we move towards start.

        if start+end too small, we move from left towards the middle
        
        do this until we find an answer
        """
        left = 0;
        right = len(numbers)-1;
        while (numbers[left] + numbers[right] != target):
            if numbers[left] + numbers[right] > target:
                right-=1;
            elif numbers[left] + numbers[right] < target:
                left+=1;
        return [left+1, right+1];

solution = Solution();
numbers = [-10,-8,-2,1,2,5,6];
target = 0;
result = solution.twoSum(numbers, target);
print(result);