package com.company;

public class SelectionSort {

    public static void swap(int[] array, int i, int j)  {
        if (i == j) {
            return;
        }
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    public static void main(String[] args)  {

        int[] intArray = {20, 35, -15, 7,  55, 1, -22};

        for (int LastUnsortedIndex = intArray.length-1; LastUnsortedIndex >0;
                LastUnsortedIndex--)    {

            int LargestIndex = 0;

            for (int i = 1; i <= LastUnsortedIndex; i++)    {
                if (intArray[LargestIndex] < intArray[i])   {
                    LargestIndex = i;

                }

            }

            swap(intArray, LargestIndex, LastUnsortedIndex);

        }


        for (int i = 0; i <intArray.length; i++)    {
            System.out.println(intArray[i]);
        }

    }
}
