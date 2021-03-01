def merge_sort(array):

    def merge(arr, start, mid, end):

        if arr[mid-1] <= arr[mid]:
            return

        i, j = start, mid
        temp_idx = 0
        temp = []

        while (i < mid) and (j < end):
            if arr[i] <= arr[j]:
                temp.append(arr[i])
                i += 1
            else:
                temp.append(arr[j])
                j += 1
            temp_idx += 1

        if i < mid:
            arr[start+temp_idx: end] = arr[i: mid]
        arr[start: start+temp_idx] = temp

    def sorting(arr, start, end):
        if (end - start) < 2:
            return

        mid = (start + end) // 2

        sorting(arr, start, mid)
        sorting(arr, mid, end)
        merge(arr, start, mid, end)

    sorting(array, 0, len(array))

    return array


print(merge_sort([20, 35, -15, 7, 55, 1, -22]))
