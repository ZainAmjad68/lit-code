class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        with sorting, we sort and then have a count that increments for every increasing number, 
        or remains same if number is same as last, 
        or resets if number is not consecutive to last
        """
        # edge case 1        
        if not nums:
            return 0;

        longestSequence = 1;
        currentSequences = [];
        sortedList = sorted(nums);
        print('sortedList',sortedList);
        previous = sortedList[0] if sortedList else None;
        for num in sortedList[1:]:
            if previous + 1 == num:
                longestSequence += 1;
            elif previous == num:
                continue;
            else:
                currentSequences.append(longestSequence);
                longestSequence = 1;
            
            currentSequences.append(longestSequence);
            previous = num;
        print(currentSequences);
        return max(currentSequences) if currentSequences else longestSequence;




solution = Solution();
nums = [-8,-4,9,9,4,6,1,-4,-1,6,8];
result = solution.longestConsecutive(nums);
print(result);