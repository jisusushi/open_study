def InsertionSort(array):
    first_unsorted_index= 1
    while first_unsorted_index < len(array):
        new_element= array[first_unsorted_index]
        new_index= first_unsorted_index

        while new_index > 0 and array[new_index-1] > new_element:
            array[new_index] = array[new_index-1]
            new_index -= 1

        array[new_index] = new_element

        first_unsorted_index += 1

    return array

print(InsertionSort([20, 35, -15, 7,  55, 1, -22]))