def SelectionSort(array):

    def swap(arr, i ,j):
        if i == j:
            pass
        else:
            temp= arr[i]
            arr[i] = arr[j]
            arr[j] = temp

        return arr

    LargestUnsortedIndex= len(array) - 1
    while LargestUnsortedIndex > 0:
        LargestIndex= 0
        for i in range(1, LargestUnsortedIndex+1):
            if array[i] > array[LargestIndex]:
                LargestIndex = i

        swap(array, LargestIndex, LargestUnsortedIndex)

        LargestUnsortedIndex-=1

    return array

print(SelectionSort([35, -22,  15, 1, 7, 20, 55]))