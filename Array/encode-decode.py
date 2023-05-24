class Solution:
    """
    @param: strs: a list of strings
    @return: encodes a list of strings to a single string.
    """
    def encode(self, strs):
        finalStr = "";
        for str in strs:
            txt = "{length}#{s}";
            finalStr += txt.format(length = len(str), s = str);
        return finalStr;

    """
    @param: str: A string
    @return: dcodes a single string to a list of strings
    """
    def decode(self, str):
        finalList = [];
        while True:
            delimiterIndex = str.find("#");
            if delimiterIndex != -1:
                lengthOfString = int(str[:delimiterIndex]);
                lengthOfInt = len(str[:delimiterIndex]);
                string = str[(lengthOfInt+1):(lengthOfString+lengthOfInt+1)];
                finalList.append(string);
                str = str[lengthOfInt+lengthOfString+1:]
            else:
                break;
        return finalList;

solution = Solution();

strs = ['i', 'am', 'happy', 'this', 'works'];
resultEncode = solution.encode(strs);
print('resultEncode',resultEncode);

str = "1#i2#am5#happy4#this5#works";
resultDecode = solution.decode(str);
print('resultDecode',resultDecode);