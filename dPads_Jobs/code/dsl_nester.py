import dsl


DSL_DICT = {
        ('atom', 'atom') : [dsl.SimpleITE, dsl.Mlp, dsl.Propensity]
}
CUSTOM_EDGE_COSTS = {
    ('atom', 'atom') : {}
}
