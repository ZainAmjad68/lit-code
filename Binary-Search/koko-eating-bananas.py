class Solution(object):
    def minEatingSpeed(self, piles, h):
        """
        :type piles: List[int]
        :type h: int
        :rtype: int
        """

        minim = 1;
        maxim = max(piles);
        res = maxim;

        while minim <= maxim:
            mid = minim + ((maxim - minim)//2);
            timeTaken = 0;
            for pile in piles:
                timeTaken += pile//mid + (pile % mid >0);

            if timeTaken < h:
                res = min(res, timeTaken)
                maxim = mid - 1;
            elif timeTaken > h:
                minim = mid + 1;
        return res;


solution = Solution();
piles, h = [312884470], 312884469;
result = solution.minEatingSpeed(piles, h);
print(result);