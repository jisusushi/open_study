def quick_sort(array):

    def partition(arr, start, end):
        pivot = arr[start]
        i, j = start, end

        while i < j:

            while (i < j) and (arr[j-1] >= pivot):
                j -= 1
            if i < j:
                j -= 1
                arr[i] = arr[j]

            while (i < j) and (arr[i+1] <= pivot):
                i += 1
            if i < j:
                i += 1
                arr[j] = arr[i]

        arr[j] = pivot
        return j

    def sort(arr, start, end):
        if end - start < 2:
            return

        pivot_index = partition(arr, start, end)
        sort(arr, start, pivot_index)
        sort(arr, pivot_index+1, end)

    sort(array, 0, len(array))

    return array


print(quick_sort([20, 35, -15, 7, 55, 1, -22]))
