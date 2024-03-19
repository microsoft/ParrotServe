from parrot.serve.prefix_matcher import GlobalPrefixMatcher


def test_prefix_matcher():
    prefix_matcher = GlobalPrefixMatcher()

    # Will not add
    prefix_matcher.add_prefix("This is a test")

    # Will add
    for i in range(GlobalPrefixMatcher._GP_THRESHOLD + 1):
        prefix_matcher.add_prefix("A" * GlobalPrefixMatcher._START_LEN + "BBB" + str(i))

    print(prefix_matcher._prefix_counter)

    query_str = "A" * GlobalPrefixMatcher._START_LEN + "BBB" + "XXX"
    pos = prefix_matcher.query_prefix(query_str)
    assert pos != -1
    print("prefix: " + query_str[:pos], "suffix: " + query_str[pos:])


if __name__ == "__main__":
    test_prefix_matcher()
