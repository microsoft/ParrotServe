from parrot.serve.prefix_matcher import PrefixMatcher


def test_prefix_matcher():
    prefix_matcher = PrefixMatcher()

    # Will not add
    prefix_matcher.add_prefix("This is a test")

    # Will add
    for i in range(PrefixMatcher._GP_THRESHOLD + 1):
        prefix_matcher.add_prefix("A" * PrefixMatcher._START_LEN + "BBB" + str(i))

    print(prefix_matcher._prefix_counter)

    query_str = "A" * PrefixMatcher._START_LEN + "BBB" + "XXX"
    pos = prefix_matcher.query_prefix(query_str)
    assert pos != -1
    print("prefix: " + query_str[:pos], "suffix: " + query_str[pos:])


if __name__ == "__main__":
    test_prefix_matcher()
