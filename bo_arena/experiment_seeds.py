"""Canonical seeds for running experiments.

Generation
----------
The seeds were derived from master seed 0 via NumPy's SeedSequence hierarchy:

    master = np.random.SeedSequence(0)
    children = master.spawn(20)
    EXPERIMENT_SEEDS = [int(child.generate_state(1)[0]) for child in children]

"""

EXPERIMENT_SEEDS: list[int] = [
    3757552657,
    673228719,
    3241444873,
    3685993406,
    1216546553,
    2078861726,
    2471122328,
    23012616,
    3031610183,
    2976135721,
    556745865,
    4118645239,
    2090914255,
    1802160804,
    3957305396,
    2416193258,
    2322685169,
    3860999861,
    3521360814,
    2710367621,
]
