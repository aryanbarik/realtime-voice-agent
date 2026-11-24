from server.chunker import ClauseChunker


def chunks(text, first=12, later=40):
    c = ClauseChunker(first, later)
    return c.push(text) + ([tail] if (tail := c.flush()) else [])


def test_sentence_chunks():
    assert chunks('Sure, I can help with that. One moment please.') == ['Sure, I can help with that.', 'One moment please.']


def test_numbers_and_abbreviation():
    assert chunks('We have 7:30 or 8.00 available, Dr. Smith.') == ['We have 7:30 or 8.00 available,', 'Dr. Smith.']


def test_long_first_clause():
    assert chunks('Certainly I can help, one moment please.') == ['Certainly I can help,', 'one moment please.']


def test_punctuation_across_deltas():
    c = ClauseChunker(12, 40)
    assert c.push('Hello there.') == []
    assert c.push(' Next') == ['Hello there.']
    assert c.flush() == 'Next'
