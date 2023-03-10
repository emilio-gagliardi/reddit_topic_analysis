import string
import os
import sys
import pytest
import tempfile
import random
import praw
from unittest.mock import Mock
from unittest import TestCase, mock
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

# Add the path to the src directory to the sys.path list
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import reddit_topic_analysis.data.data_processing as sut


def acceptable_chars() -> str:
    return string.ascii_letters + string.digits + '_!@#$%^&*()'


@pytest.fixture
def mock_spacy_doc(request):
    text = request.param
    doc = Mock(spec=spacy.tokens.doc.Doc)
    doc.__iter__.return_value = iter([Mock(text=word, is_alpha=True, spec=spacy.tokens.Token) for word in text.split()])
    doc.__len__.return_value = len(text.split())
    return Mock(return_value=doc, spec=spacy.tokens.doc.Doc)


@pytest.fixture
def sample_text():
    sample = '''Pandas is an industry standard for analyzing data in Python. 
    With a few keystrokes, you can load, filter, restructure, and 
    visualize gigabytes of heterogeneous information.'''
    return sample


class TestCountWords(TestCase):
    def test_count_words_len_only(self, mock_spacy_doc, sample_text):
        pass

    def test_count_words_correct_tokens(self):
        pass
