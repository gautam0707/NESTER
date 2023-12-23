import dsl


CUSTOM_EDGE_COSTS = {
    ('atom', 'atom') : {}
}
DSL_DICT = {
        ('atom', 'atom') : [ dsl.Mlp, dsl.Align, dsl.Propensity]
}