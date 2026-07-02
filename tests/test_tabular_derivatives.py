import pandas as pd

from neuroalign.data.loaders.tabular_derivatives import _NETWORK_PREFIX_RE


def test_network_prefix_stripped_matches_diffusion_naming():
    """Anatomical labels must normalize to the same names diffusion already uses."""
    anat_labels = pd.Series(["7Networks_LH_Cont_Cing_1", "17Networks_RH_Vis_2", "LH-Thal"])
    normalized = anat_labels.str.replace(_NETWORK_PREFIX_RE, "", regex=True)

    assert list(normalized) == ["LH_Cont_Cing_1", "RH_Vis_2", "LH-Thal"]

    diffusion_labels = {"LH_Cont_Cing_1", "RH_Vis_2"}
    assert diffusion_labels.issubset(set(normalized))
