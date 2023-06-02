class MinStack(object):

    def __init__(self):
        self.stack = [];
        self.minStack = [];        

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.stack.append(val);
        # whenever we push something, add a new value to minStack detailing the new min as well 
        self.minStack.append(min(val, self.minStack[-1] if self.minStack else val));        

    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop();
        # pop minStack too so that we get the previous min value on top now.
        self.minStack.pop();        

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1];
        

    def getMin(self):
        """
        :rtype: int
        """
        # return whatever is at top of minStack
        return self.minStack[-1];
        


# Your MinStack object will be instantiated and called as such:
obj = MinStack()
obj.push(2);
obj.push(1);
obj.push(8);
obj.pop();
obj.push(5);
print(obj.top());
print(obj.getMin());