class TimeMap(object):

    def __init__(self):
        self.store = {};

    def set(self, key, value, timestamp):
        """
        :type key: str
        :type value: str
        :type timestamp: int
        :rtype: None
        """
        if (key not in self.store):
            self.store[key] = [];
        arr = self.store[key];
        arr.append([value, timestamp]);

        print('store now:',self.store);


    def get(self, key, timestamp):
        """
        :type key: str
        :type timestamp: int
        :rtype: str
        """
        print("getting",key,"at time:",timestamp,'from store');
        print('store is ',self.store);

        if (key not in self.store):
            return None;
        keyToCheck = self.store[key];
        res = "";

        low, high = 0, len(keyToCheck) - 1;
        while low <= high:
            mid = low + ((high - low) // 2)
            if keyToCheck[mid][1] > timestamp:
                high = mid - 1;
            elif keyToCheck[mid][1] < timestamp:
                low = mid + 1;
                res = keyToCheck[mid][0];
            else:
                return keyToCheck[mid][0];
        return res;

# Your TimeMap object will be instantiated and called as such:
obj = TimeMap()

#
obj.set('foo','bar',1);
obj.set('foo','bar2',4);

res = obj.get('foo', 5);
print(res);
# param_2 = obj.get(key,timestamp)