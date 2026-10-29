from utils import autocomplete as ac


def test_manufacturer_prefixes():
    assert ac.suggest_manufacturers("Man")[0] == "Mankind Pharma"
    assert ac.suggest_manufacturers("Sun")[0] == "Sun Pharma"
    assert ac.suggest_manufacturers("Cip")[0] == "Cipla"


def test_case_insensitive_and_word_start():
    assert "Dr. Reddy's Laboratories" in ac.suggest_manufacturers("redd")
    assert "Glenmark Pharmaceuticals" in ac.suggest_manufacturers("glen")


def test_unknown_returns_empty():
    assert ac.suggest_manufacturers("Zzqx") == []


def test_dependent_lookup():
    e = ac.lookup_medicine("dolo 650")
    assert e and e.manufacturer == "Micro Labs" and e.composition == "Paracetamol"
    assert ac.lookup_medicine("Totally Unknown") is None
    assert "Manforce 50" in ac.medicines_for_manufacturer("Mankind Pharma")
