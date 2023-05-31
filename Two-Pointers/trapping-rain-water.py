class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        Brute Force:
        - for every bar in the height array, calculate the maximum height bar to the left and right.
        - subtract the height of the current bar from the minimum of the left and right maximum bar heights to get the amount of water that can be trapped.
        - add the result to a variable "water" which keeps track of the total amount of water that can be trapped.
        - return the value of the "water" variable.

        Optimized (this one):
        - initialize left, right pointers to the first and last bars of the height array, respectively.
        - initialize variables left_max and right_max to zero.
        - while the left pointer is less than or equal to the right pointer, compare the heights of the bars pointed to by the left and right pointers.
        - if the height of the left bar is less than or equal to the height of the right bar, check if the height of the left bar is greater than the left_max variable. If it is, update left_max, otherwise, add left_max - height[left] to the "water" variable. Move the left pointer to the next bar.
        - if the height of the right bar is less than the height of the left bar, check if the height of the right bar is greater than the right_max variable. If it is, update right_max, otherwise, add right_max - height[right] to the "water" variable. Move the right pointer to the previous bar.
        - return the value of the "water" variable.
        """
        if not height:
            return 0;
    
        water = 0;
        l,r = 0, len(height) - 1;
        maxLeft, maxRight = height[l], height[r];

        while l < r:
            if maxLeft < maxRight:
                l += 1;
                water += 0 if maxLeft - height[l] < 0 else maxLeft - height[l];
                maxLeft = max(maxLeft, height[l]);
            else:
                r -= 1;
                water += 0 if maxRight - height[r] < 0 else maxRight - height[r];
                maxRight = max(maxRight, height[r]);
        return water;

solution = Solution();
height = [0,1,0,2,1,0,1,3,2,1,2,1];
result = solution.trap(height);
print(result);

