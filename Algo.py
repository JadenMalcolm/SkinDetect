def merge_and_count(arr, temp_arr, left, mid, right):
    i = left    # Starting index for left subarray
    j = mid + 1 # Starting index for right subarray
    k = left    # Starting index to be sorted
    merge_inv_count = 0

    # Count inversions during merging
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            # All remaining elements in left subarray (arr[i:mid]) are greater than arr[j]
            merge_inv_count += (mid - i + 1)
            temp_arr[k] = arr[j]
            j += 1
        k += 1

    # Copy the remaining elements of left subarray, if any
    while i <= mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1

    # Copy the remaining elements of right subarray, if any
    while j <= right:
        temp_arr[k] = arr[j]
        j += 1
        k += 1

    # Copy the sorted subarray into Original array
    for i in range(left, right + 1):
        arr[i] = temp_arr[i]

    return merge_inv_count


def merge_sort_and_count(arr, temp_arr, left, right):
    if left < right:
        mid = (left + right) // 2

        # Count inversions in left subarray
        left_inv_count = merge_sort_and_count(arr, temp_arr, left, mid)

        # Count inversions in right subarray
        right_inv_count = merge_sort_and_count(arr, temp_arr, mid + 1, right)

        # Count inversions during the final merge
        merge_inv_count = merge_and_count(arr, temp_arr, left, mid, right)

        return left_inv_count + right_inv_count, merge_inv_count

    return 0, 0


def count_inversions(arr):
    temp_arr = [0] * len(arr)
    left_inv_count, merge_inv_count = merge_sort_and_count(arr, temp_arr, 0, len(arr) - 1)
    return left_inv_count, merge_inv_count


arr = [14, 9, 13, 10, 15, 12, 6, 8]
left_count, final_merge_count = count_inversions(arr)

print("Count of Inversions in Left:", left_count)
print("Count of Inversions in Final Merge:", final_merge_count)
