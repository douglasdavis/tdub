from tdub.regions import Region


def test_from_str():
    assert Region.from_str("2j2b") == Region.r2j2b
    assert Region.from_str("1j1b") == Region.r1j1b
    assert Region.from_str("2j1b") == Region.r2j1b

    assert Region.from_str("r2j2b") == Region.r2j2b
    assert Region.from_str("r1j1b") == Region.r1j1b
    assert Region.from_str("r2j1b") == Region.r2j1b

    assert Region.from_str("reg2j2b") == Region.r2j2b
    assert Region.from_str("reg1j1b") == Region.r1j1b
    assert Region.from_str("reg2j1b") == Region.r2j1b
