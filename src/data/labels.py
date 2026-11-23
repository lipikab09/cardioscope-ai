AAMI_MAP={"N":"N","L":"N","R":"N","e":"N","j":"N","A":"S","a":"S","J":"S","S":"S","V":"V","E":"V","F":"F","/":"Q","f":"Q","Q":"Q"}
CLASSES=["N","S","V","F","Q"]
CLASS_TO_IDX={c:i for i,c in enumerate(CLASSES)}
def map_symbol(symbol): return AAMI_MAP.get(symbol)
