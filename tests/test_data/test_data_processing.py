import reddit_topic_analysis.data.data_processing


def test_connect_to_reddit():
    assert reddit_topic_analysis.data.data_processing.connect_to_reddit_with_oauth() == True
