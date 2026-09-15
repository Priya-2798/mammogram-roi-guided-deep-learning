import pandas as pd

from src.multimodal.encoding import encode, fit_encoder


def test_unknown_category_mapping():
    train = pd.DataFrame({"mass_shape": ["ROUND", "OVAL"]})
    test = pd.DataFrame({"mass_shape": ["IRREGULAR"]})
    cats = fit_encoder(train, ["mass_shape"])
    encoded = encode(test, cats, ["mass_shape"])
    unknown_index = cats["mass_shape"].index("UNKNOWN")
    assert encoded[0, unknown_index] == 1.0
