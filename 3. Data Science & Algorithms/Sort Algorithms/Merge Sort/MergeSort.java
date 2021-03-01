package com.company;

public class MergeSort {

    public static void merge(int[] input, int start, int mid, int end)  {

        if (input[mid-1] <= input[mid]) {       // Simple Optimization: when the input array is already sorted
            return;
        }

        int i = start;
        int j = mid;
        int tempIndex= 0;
        int[] temp = new int[end-start];

        while (i < mid && j < end)  {
            temp[tempIndex++] = input[i] <= input[j] ? input[i++] : input[j++];     // simple method replacing 'if'
        }

        // Simple Optimization: don't have to do anything if there are leftover in right array
        System.arraycopy(input, i, input, start+tempIndex, mid-i);  // powerful method 'arraycopy'
        System.arraycopy(temp, 0,input, start, tempIndex);

    }

    public static void mergeSort(int[] input, int start, int end)   {

        if (end - start < 2)    {       // Recursive method - need escape condition
            return;
        }

        int mid = (start + end) / 2;

        mergeSort(input, start, mid);
        mergeSort(input, mid, end);
        merge(input, start, mid, end);

    }

    public static void main(String[] args)  {
        int[] intArray= {20, 35, -15, 7, 55, 1, -22};

        mergeSort(intArray, 0, intArray.length);    // end: not the last index, but +1!!!

        for (int i = 0; i<intArray.length; i++) {
            System.out.println(intArray[i]);
        }
    }

}
