

# once we know that we can't extend the area of a bar any further (i.e.; bars are in decreasing trajectory towards the right), we pop it and see if its area was higher than the max one so far, if so, assign it to max.
# since we only pop the most recent bars and then go from there, a stack is a good Data Structure to use.

class Solution(object):
    def largestRectangleArea(self, heights):
        maxArea = 0;
        stack = []; # pair of (index, height)

        for i,h in enumerate(heights):
            start = i;
            if (stack and stack[-1][1] > h):  # if the height of top of stack is greater than the height we're at
                index, height = stack.pop();
                maxArea = max(maxArea, h*(i-index)); # width is current index (i) - where that element started (index)
                start = index; # because this height was greater than the current height we're visiting so we can extend
            stack.append((start, h));

        # for elements that remained in the stack till the end i.e.; they extend over the whole histogram
        for i,h in stack:
            maxArea = max(maxArea, h*(len(heights)-1)); # calculate their area as well
        
        return maxArea;
