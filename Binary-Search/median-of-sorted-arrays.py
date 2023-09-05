class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        A, B = nums1, nums2;
        if len(B) < len(A):
            B, A = nums1, nums2;

        total = len(A) + len(B);
        half = total // 2;

        l, r = 0, len(A)-1
        while True:
            i = (l+r) // 2;
            j = (half - (i+1) - 1)

            Aleft = A[i];
            Aright = A[i+1];
            Bleft = B[j];
            Bright = B[j+1];
            
            if Aleft <= Bright and Bleft <= Aright:
                if total % 2:
                    return min(Aright, Bright)
                return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            elif Aleft > Bright:
                r = i - 1;
            else:
                l = i + 1;
