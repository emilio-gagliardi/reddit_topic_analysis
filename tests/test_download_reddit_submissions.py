import reddit_topic_analysis.data.data_processing as sut


def test_connect_to_reddit_with_oauth():
    reddit = sut.praw.Reddit(client_id=sut.os.environ['REDDIT_CLIENT_ID'],
                             client_secret=sut.os.environ['REDDIT_CLIENT_SECRET'],
                             redirect_uri='http://localhost:8000',
                             user_agent='bhg_topic_analysis_trends')
    assert isinstance(reddit, sut.praw.Reddit)


def test_get_one_subreddit():
    pass


def test_extract_submission_info():
    pass


def test_dicts_to_json():
    pass


def test_save():
    # one of the following should be true
    success = save_as_documents()
    success = save_as_tables()
    pass


def test_save_as_documents():
    pass


def test_save_as_tables():
    pass
