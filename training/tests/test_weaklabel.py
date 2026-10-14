from tv.data.weaklabel import weak_label


def test_failing_keyword():
    assert weak_label("3D print FAILED - spaghetti everywhere") == "failing"


def test_normal_keyword():
    assert weak_label("Perfect 3D print timelapse") == "normal"


def test_failing_takes_precedence():
    assert weak_label("timelapse of a failed print") == "failing"


def test_no_signal_is_uncertain():
    assert weak_label("Unboxing a new printer") == "uncertain"


def test_case_insensitive():
    assert weak_label("LAYER SHIFT disaster") == "failing"
