def BubbleSort(array):
    LastUnsortedIndex= len(array)-1

    def swap(arr, i ,j):
        if i != j:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp

        return arr

    while LastUnsortedIndex > 0:
        for i in range(0, LastUnsortedIndex):
            if array[i] > array[i+1]:
                swap(array, i, i+1)

        LastUnsortedIndex -= 1

    return array

print(BubbleSort([20, 35, -15, 7, 55, 1, -22]))