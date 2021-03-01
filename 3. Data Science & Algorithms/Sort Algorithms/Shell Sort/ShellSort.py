def ShellSort(array):
    gap = len(array) // 2
    while gap > 0:
        for searching_index in range(gap, len(array)):
            new_element= array[searching_index]
            new_index= searching_index

            while new_index >= gap and array[new_index-gap] > new_element:
                array[new_index] = array[new_index - gap]
                new_index -= gap

            array[new_index] = new_element

        gap //= 2

    return array

print(ShellSort([20, 35,-15, 7, 55, 1, -22]))