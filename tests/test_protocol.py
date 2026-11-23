from src.data.splits import TRAIN,VAL,DS2,validate_splits
from src.data.labels import map_symbol
def test_disjoint(): assert validate_splits() and not(set(TRAIN)&set(VAL)) and not(set(TRAIN)&set(DS2)) and not(set(VAL)&set(DS2))
def test_paired_subject_excluded(): assert "201" in TRAIN and "202" not in TRAIN+VAL+DS2
def test_labels():
    assert [map_symbol(x) for x in ["N","A","V","F","/","+"]]==["N","S","V","F","Q",None]
