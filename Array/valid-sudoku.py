class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """

        for index, row in enumerate(board):
            checkerRow = {};
            checkerColumn = {};
            # to check the rows
            for box in row:
                print('rowBox',box);
                print('checkerRow',checkerRow);
                if box != ".":
                    if box not in checkerRow:
                        checkerRow[box] = 1;
                    else:
                        return False;
        
            # to check the columns
            for x in range(len(row)):
                print('columnBox',board[x][index]);
                print('checkerColumn',checkerColumn);

                if board[x][index] != ".":
                    if board[x][index] not in checkerColumn:
                        checkerColumn[board[x][index]] = 1;
                    else:
                        return False;

        # to check 3*3 sub-boxes
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                submatrix = []
                for k in range(i, i + 3):  # Iterate over rows of the submatrix
                    submatrix.append(board[k][j:j + 3])  # make a 3*3 sub-matrix
                # flatten the sub-matrix and iterate over it
                flattened = [item for sublist in submatrix for item in sublist];
                checkerSubBox = {};
                for subbox in flattened:
                    if subbox != ".":
                        if subbox not in checkerSubBox:
                            checkerSubBox[subbox] = 1;
                        else:
                            return False;
        return True;

solution = Solution();
board = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]];
result = solution.isValidSudoku(board);
print(result);