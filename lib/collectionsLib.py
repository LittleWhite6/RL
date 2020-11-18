'''namedtuple('名称', [属性list])'''
from collections import namedtuple

Point = namedtuple('Point',['x','y'])
p = Point(1,2)

print(p.x,p.y)

DensenetParams = namedtuple('DensenetParameters', ['num_classes',
                                         'first_output_features',
                                         'layers_per_block',
                                         'growth_rate',
                                         'bc_mode',
                                         'is_training',
                                         'dropout_keep_prob'
                                         ])

default_params = DensenetParams(
        num_classes = 10,
        first_output_features = 24,
        layers_per_block = 12,
        growth_rate = 12,
        bc_mode = False,
        is_training = True,
        dropout_keep_prob = 0.8,
        )

print(default_params.num_classes)