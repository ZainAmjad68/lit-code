class Solution(object):
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        Naive solution is a nested loop,
        where for each element, we find the index that has a value greater 
        than that element and see how far away that value is 
        (index where such value is found - current index)
        """
        result = [];
        for currentIndex, current in enumerate(temperatures):
            for futureIndex, future in enumerate(temperatures, currentIndex+1):
                if futureIndex < len(temperatures) and temperatures[futureIndex] > temperatures[currentIndex]:
                    result.append(futureIndex-currentIndex);
                    break;
                
                if futureIndex >= len(temperatures):
                    result.append(0);
                    break;
        return result;

solution = Solution();
temperatures = [73,74,75,71,69,72,76,73];
res = solution.dailyTemperatures(temperatures);
print(res);