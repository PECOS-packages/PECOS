"""Module for syndrome difference calculations."""


def syn_diff(
    data: dict[str, list[str]],
    pairs: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Performs bitwise XOR between corresponding binary strings from specified pairs.

    Args:
        data: Dictionary with string keys and list of binary strings as values
        pairs: List of tuples specifying which keys to compare

    Returns:
        Dictionary with keys like "key1_key2" and values as lists of XOR results
    """
    result = {}

    for key1, key2 in pairs:
        # Create the new key name
        new_key = f"{key1}_{key2}"

        # Get the lists of binary strings
        list1 = data[key1]
        list2 = data[key2]

        # Perform XOR on corresponding strings
        xor_results = []
        for str1, str2 in zip(list1, list2, strict=False):
            # Convert binary strings to integers, XOR them, then back to binary string
            int1 = int(str1, 2)
            int2 = int(str2, 2)
            xor_result = int1 ^ int2

            # Convert back to binary string with same length as original
            xor_binary = format(xor_result, f"0{len(str1)}b")
            xor_results.append(xor_binary)

        result[new_key] = xor_results

    return result
