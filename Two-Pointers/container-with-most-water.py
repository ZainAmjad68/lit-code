class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        in order to find maximum contained water, we begin with two pointers from start and end
        and find the bottleneck i.e.; the one with lower height (xheight), as well as the current width
        then we see if their product is bigger than the biggest area calculated yet
        we also update pointer by looking at which of the two has lesser height
        """
        maxH = 0; 
        left = 0;
        right = len(height) - 1;
        while left < right:
            xheight = min(height[left], height[right]);
            width = right - left;
            maxH = max(maxH, xheight * width);

            if (height[left] < height[right]):
                left+=1;
            else:
                right-=1;
        return maxH;

solution = Solution();
height = [1,8,6,2,5,4,8,3,7];
result = solution.maxArea(height);
print(result);

