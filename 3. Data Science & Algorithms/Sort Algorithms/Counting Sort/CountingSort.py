def counting_sort(array, max_, min_):

    counting_array = [0 for i in range(max_ - min_ + 1)]

    for val in array:
        counting_array[val - min_] += 1

    idx = 0
    for count_idx, count in enumerate(counting_array):
        while count > 0:
            array[idx] = min_ + count_idx
            count -= 1
            idx += 1

    return array


print(counting_sort([2, 5, 9, 8, 2, 8, 7, 10, 4, 3], 10, 1))
