class Solution(object):
    def carFleet(self, target, position, speed):
        """
        :type target: int
        :type position: List[int]
        :type speed: List[int]
        :rtype: int
        """

        stack = [];
        pair = [[p,s] for p,s in zip(position, speed)];
        pair.sort(reverse=True);
        print('sorted pair', pair);

        for pos,spe in pair:
            print('position', pos);
            print('speed', spe);
            print('time to reach', (target - pos)/spe);
            stack.append((target - pos)/spe);
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop();

        print('stack at the end', stack);
        return len(stack);

solution = Solution();
target = 12;
position = [10,8,0,5,3];
speed = [2,4,1,1,3];
result = solution.carFleet(target, position, speed);
print(result);