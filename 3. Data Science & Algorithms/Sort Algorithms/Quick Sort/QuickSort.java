package com.company;

public class QuickSort {

    public static int partition(int[] input, int start, int end) {
        int pivot = input[start];
        int i = start;
        int j = end;

        while (i < j)   {

            // Empty Loop body - for decrementing j
            while (i < j && input[--j] >= pivot);   // Prefix Decrement: j will be decremented first
            if (i < j)  {
                input[i] = input[j];    // Index i is already preserved (in 'pivot' or right side)
            }

            while (i < j && input[++i] <= pivot);
            if (i < j)  {
                input[j] = input[i];
            }
        }

        input[j] = pivot;
        return j;

    }

    public static void quickSort(int[] input, int start, int end)   {
        if (end - start < 2)    {
            return;
        }

        int pivotIndex = partition(input, start, end);
        quickSort(input,  start, pivotIndex);
        quickSort(input, pivotIndex+1, end);

    }

    public static void main(String[] args)  {
        int[] intArray = {20, 35, -15, 7, 55, 1, -22};

        quickSort(intArray, 0, intArray.length);

        for (int i = 0; i < intArray.length; i++)   {
            System.out.println(intArray[i]);

        }
    }


}
