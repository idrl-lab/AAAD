from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

# eval-EXP-20211204-074723
DARTS_V3 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1),('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
PDARTS = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


# PCDARTS = PC_DARTS_cifar

# eval-EXP-20211108-110107
PCDARTS_1 = Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

# eval-EXP-20211112-085115
PCDART_FGSM_1_100 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

# eval-EXP-20211114-021311
PCDARTS_FGSM_es_small = Genotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6)) #0.01

# eval-EXP-20211130-191653
# PCDARTS_natural = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
# DARTS_natural = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

#eval-EXP-20211201-225407
PCDARTS_caa = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

# eval-EXP-20211205-022515
PCDARTS_CAA_1 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# eval-EXP-20211205-153001
PCDARTS_CAA_es = Genotype(normal=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 4), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
PCDARTS = PCDARTS_CAA_es

PCDARTS_CAA_noise_FGSM = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

ES_CRNA_ME = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('skip_connect', 1), ('sep_conv_5x5', 4), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))
# eval-EXP-20211109-024423
# PCDARTS = PCDARTS_Jacobian
# PCDARTS = DARTS_V1
# eval-CAA_noise_PGD-20211207-081059
# PCDARTS_CAA_noise_PGD = genotype = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
#
# # eval-DARTS_Arch0-20211208-022638
# DARTS_Arch0 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))
#
# PCDARTS_Diff_PGD = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
#
# NASP_Jacobian = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
#
#
# PCDARTS_FGSM_1_10 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
#
# PCDARTS_FGSM_1_10_skip = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
#
# PCDARTS_FGSM_1_10_wopool = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
# PCDARTS_PGD_noise = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 3), ('dil_conv_5x5', 3), ('skip_connect', 1)], reduce_concat=range(2, 6))
#
# DARTS_FGSM_1_100 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))
#
# PC_DARTS_PGD = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
#
# DARTS_PGD_1_100 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
#
# PC_DARTS_FGSM_es_pool = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('max_pool_3x3', 2), ('dil_conv_5x5', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
#
# PC_DARTS_FGSM_es_pool_2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('skip_connect', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
#
# # early stopping处理之后
# DARTS_PGD_1_10_pool2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
#
# DARTS_FGSM_1_100_pool2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
#
# DARTS_FGSM_1_10_pool2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
#
# DARTS_PGD_1_100_pool2 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
#
# PCDARTS_PGD_1_10_pool2 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('max_pool_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
#
# PC_DARTS_PGD_1_100_pool2 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
#
# PCDARTS_FGSM_1_10_pool2 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
#
# NASP_FGSM_1_10 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))
#
# NASP_FGSM_1_100 = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 4), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('max_pool_3x3', 1), ('sep_conv_3x3', 4), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
#
# NASP_PGD_100 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
#
# NASP_PGD_1_10 = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))
#
# PC_DARTS_CAA_1_100_pool2 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
#
# DARTS_CAA_1_100_pool2 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
#
# DARTS_CAA_1_100_pool2_skip = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
#
# NASP_CAA_1_100_pool2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
#
# NASP_CAA_1_100_pool2_skip = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('skip_connect', 2), ('sep_conv_3x3', 3), ('skip_connect', 3), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
#
# DARTS_CAA_1_10_pool2 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
#
# NASP_CAA_1_10_pool2 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
#
# PC_DARTS_CAA_1_10_pool2 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
#
# ### darts
# DARTS_pgd_1_10 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
#
# DARTS_fgsm_1_10 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
#
# DARTS_fgsm_1_100 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 4), ('skip_connect', 3)], reduce_concat=range(2, 6))
#
# DARTS_pgd_1_100 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
#
#
# darts_fgsm_1_100 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
#
# darts_fgsm_1_10 =  Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
#
# darts_pgd_1_100 =  Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
#
# dual_adv_fgsm = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

# DARTS_total = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
#
# advrush = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# # RESTART

DARTS_natural = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

DARTS_System = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

DARTS_Jacobian = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('max_pool_3x3', 3), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('skip_connect', 0), ('max_pool_3x3', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

DARTS_FGSM = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

DARTS_Hessian = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

DARTS_PGD = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

DARTS_Clean = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))


PCDARTS_FGSM = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

PCDARTS_System = Genotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

PCDARTS_natural = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('skip_connect', 3), ('skip_connect', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))

PCDARTS_Clean = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

PCDARTS_Hessian = Genotype(normal=[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

PCDARTS_Jacobian = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

PCDARTS_PGD = Genotype(normal=[('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 4), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

NASP_Clean = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('skip_connect', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

NASP_System = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 0), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))

NASP_FGSM = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

NASP_PGD = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('dil_conv_5x5', 0)], reduce_concat=range(2, 6))

NASP_Jacobian = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

NASP_Hessian = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

NASP_natural = Genotype(normal=[('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

FairDARTS_Clean = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

FairDARTS_FGSM = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

FairDARTS_natural = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 4), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 1), ('dil_conv_3x3', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

FairDARTS_PGD = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

FairDARTS_System = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('skip_connect', 1), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

FairDARTS_Jacobian = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('max_pool_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

FairDARTS_Hessian = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 3), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

SmoothDARTS_Clean = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 3), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

SmoothDARTS_natural = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('dil_conv_5x5', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))

SmoothDARTS_System = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))

SmoothDARTS_FGSM = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))

SmoothDARTS_PGD = Genotype(normal=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 4), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

SmoothDARTS_Jacobian = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))

SmoothDARTS_Hessian = genotype = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

Robust_MP = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

Robust_MP_2 = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))

Robust_MP_3 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 3)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 3), ('skip_connect', 2), ('dil_conv_3x3', 3)], reduce_concat=[2,3,4,5])

Random_search_PGD = Genotype(normal=[('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

Random_search_Clean = Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

Random_search_FGSM = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

Random_search_natural = Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

Random_search_System = Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

Random_search_Jacobian = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('skip_connect', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 3), ('max_pool_3x3', 3), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

Random_search_Hessian = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

NASNet_Random  = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

Random_search_weight_sharing_System = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('avg_pool_3x3', 1), ('sep_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 3), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 3)], reduce_concat=[2,3,4,5])

Random_search_weight_sharing_natural = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0), ('max_pool_3x3', 1)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)], reduce_concat=[2,3,4,5])

Random_search_weight_sharing_FGSM = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 4), ('max_pool_3x3', 3)], reduce_concat=[2,3,4,5])

Random_search_weight_sharing_PGD = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('dil_conv_5x5', 4), ('skip_connect', 0)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('sep_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=[2,3,4,5])

Random_search_weight_sharing_Jacobian = Genotype(normal=[('none', 1), ('none', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('none', 1), ('skip_connect', 0), ('avg_pool_3x3', 4), ('avg_pool_3x3', 2)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 0), ('none', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2,3,4,5])

Random_search_weight_sharing_Hessian = Genotype(normal=[('none', 1), ('none', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('none', 1), ('skip_connect', 0), ('avg_pool_3x3', 4), ('avg_pool_3x3', 2)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 0), ('none', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 4)], reduce_concat=[2,3,4,5])

DE_natural = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('avg_pool_3x3', 3)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('dil_conv_5x5', 3), ('skip_connect', 4), ('avg_pool_3x3', 1)], reduce_concat=[2,3,4,5])

DE_System = Genotype(normal=[('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_5x5', 3)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 4)], reduce_concat=[2,3,4,5])

DE_FGSM = Genotype(normal=[('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('skip_connect', 0), ('avg_pool_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('skip_connect', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=[2,3,4,5])

DE_Jacobian = Genotype(normal=[('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('skip_connect', 1), ('max_pool_3x3', 4)], reduce_concat=[2,3,4,5])

DE_PGD = Genotype(normal=[('dil_conv_5x5', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('skip_connect', 4)], reduce_concat=[2,3,4,5])

DE_Hessian = Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('sep_conv_3x3', 3), ('avg_pool_3x3', 4), ('skip_connect', 1)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('skip_connect', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3), ('skip_connect', 2), ('dil_conv_3x3', 3)], reduce_concat=[2,3,4,5])



#LAS-DARTS
Fix_DARTS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))
LAS_DARTS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

RACL = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3',0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5',1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
ADVRUSH = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

RNAS_H = Genotype(
  normal=[
    ('dil_conv_5x5', 0),
    ('dil_conv_5x5', 1),
    ('dil_conv_5x5', 0),
    ('dil_conv_3x3', 1),
    ('dil_conv_5x5', 3),
    ('sep_conv_5x5', 0),
    ('dil_conv_5x5', 0),
    ('dil_conv_3x3', 2)],
  normal_concat=range(2, 6),
  reduce=[
    ('avg_pool_3x3', 1),
    ('dil_conv_5x5', 0),
    ('dil_conv_5x5', 2),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 3),
    ('skip_connect', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_5x5', 4)],
  reduce_concat=range(2, 6))

RNAS_L = Genotype(
  normal=[
    ('dil_conv_5x5', 0),
    ('dil_conv_5x5', 1),
    ('skip_connect', 2),
    ('dil_conv_5x5', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 1),
    ('avg_pool_3x3', 0),
    ('skip_connect', 2)],
  normal_concat=range(2, 6),
  reduce=[
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 1),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 1),
    ('dil_conv_3x3', 3),
    ('dil_conv_5x5', 2),
    ('sep_conv_5x5', 3),
    ('sep_conv_3x3', 2)],
  reduce_concat=range(2, 6))

ES_CRA_ME = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2)], reduce_concat=[2,3,4,5])

ES_CRA_ME_1 = Genotype(normal=[('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 4), ('skip_connect', 3)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1), ('skip_connect', 3), ('sep_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_5x5', 1)], reduce_concat=[2,3,4,5])

ES_CRA_ME_2 = Genotype(normal=[('dil_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 2), ('dil_conv_5x5', 1)], normal_concat=[2,3,4,5], reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('avg_pool_3x3', 3), ('sep_conv_5x5', 2), ('dil_conv_3x3', 4), ('avg_pool_3x3', 3)], reduce_concat=[2,3,4,5])

ES_CRA_ME_3 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 4)], normal_concat=[2,3,4,5], reduce=[('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 3), ('dil_conv_3x3', 3)], reduce_concat=[2,3,4,5])

ES_CRA_ME_4 = Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 4), ('avg_pool_3x3', 1)], reduce_concat=[2,3,4,5])

ES_CRA_ME_5 = Genotype(normal=[('skip_connect', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_3x3', 2)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=[2,3,4,5])

ES_CRA_ME_6 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 1), ('max_pool_3x3', 4), ('sep_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3), ('skip_connect', 2)], reduce_concat=[2,3,4,5])

ES_CRA_ME_7 = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3)], normal_concat=[2,3,4,5], reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('avg_pool_3x3', 2), ('skip_connect', 3)], reduce_concat=[2,3,4,5])

# Random
ES_CRA_ME_8 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('avg_pool_3x3', 4)], normal_concat=[2,3,4,5], reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 4)], reduce_concat=[2,3,4,5])

JiaNet = Genotype(
    normal=[
        ('sep_conv_7x7', 0),
        ('dil_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('dil_conv_3x3', 2),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 4)
    ],
    normal_concat=[3, 5, 6],
    reduce=[
        ('sep_conv_7x7', 0),
        ('dil_conv_5x5', 1),
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 0),
        ('dil_conv_3x3', 3),
        ('avg_pool_3x3', 2),
        ('max_pool_3x3', 2),
        ('dil_conv_5x5', 2),
        ('sep_conv_7x7', 3),
        ('avg_pool_3x3', 2)
    ],
    reduce_concat=[4, 5, 6]
)

# ES-CRA-ME
ES_CRA_ME_9 = Genotype(
  normal=[
    ('dil_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('dil_conv_5x5', 1),
    ('dil_conv_3x3', 1),
    ('dil_conv_5x5', 3),
    ('dil_conv_3x3', 0),
    ('dil_conv_5x5', 2),
    ('dil_conv_3x3', 0)],
  normal_concat=range(2, 6),
  reduce=[
    ('avg_pool_3x3', 0),
    ('sep_conv_5x5', 1),
    ('dil_conv_5x5', 2),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 3),
    ('skip_connect', 1),
    ('sep_conv_5x5', 4),
    ('dil_conv_5x5', 0)],
  reduce_concat=range(2, 6))

ES_CRA_ME_10 = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 4)], normal_concat=[2,3,4,5], reduce=[('skip_connect', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4), ('avg_pool_3x3', 4)], reduce_concat=[2,3,4,5])

ES_CRA_ME_100 = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))