class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        
        Algo:
        - First Check if the target even exists in the matrix, by checking start and end of the matrix
        - Then, find the row where the target might exist
        - Run Binary Search on that Row
        """

        # check if it exists in matrix
        noOfRows = len(matrix) - 1;
        if matrix and (target < matrix[0][0] or target > matrix[len(matrix) - 1][len(matrix[noOfRows])-1]):
            return False;

        # figure out the row where it might exist
        rowToSearch = None;
        for index, row in enumerate(matrix):
            if row[0] <= target and row[len(row)-1] >= target:
                rowToSearch = index;
                break;

        # run binary search on that row
        row = matrix[rowToSearch];
        l,r = 0, len(row)-1;

        while l <= r:
            m = l + ((r-l)//2);
            if row[m] < target:
                l = m+1;
            elif row[m] > target:
                r = m-1;
            else:
                return True;
        return False;


solution = Solution();
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]; 
target = 3;
result = solution.searchMatrix(matrix, target);
print(result);