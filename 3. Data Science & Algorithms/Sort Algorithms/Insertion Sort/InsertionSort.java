package com.company;

public class InsertionSort {

    public static void main(String[] args)  {

        int[] intArray = {20, 35, -15, 7,  55, 1, -22};

        for (int firstUnsortedIndex = 1; firstUnsortedIndex < intArray.length; firstUnsortedIndex++)    {

            int newElement = intArray[firstUnsortedIndex];
            int newIndex;

            for (newIndex = firstUnsortedIndex; newIndex > 0 && intArray[newIndex-1] > newElement; newIndex--)    {
                intArray[newIndex] = intArray[newIndex-1];
            }

            intArray[newIndex] = newElement;

        }

        for (int i = 0; i < intArray.length; i++)   {
            System.out.println(intArray[i]);
        }

    }

}
