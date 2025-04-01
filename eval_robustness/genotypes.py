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
    normal=[
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
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
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
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
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
    normal_concat=[4, 5, 6],
    reduce=[
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
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

# eval-EXP-20211204-074723
random = Genotype(
    normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('max_pool_3x3', 0),
            ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

PC_DARTS_cifar = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(
    normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

# PCDARTS = PC_DARTS_cifar

# eval-PCDARTS_Clean-20211108-110107
PCDARTS_1 = Genotype(
    normal=[('skip_connect', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1),
            ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),
            ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
PCDARTS_FGSM = Genotype(
    normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 0),
            ('skip_connect', 3), ('avg_pool_3x3', 0), ('max_pool_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

# eval-EXP-20211112-085115
PCDART_FGSM_es = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0),
            ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('sep_conv_5x5', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0),
            ('sep_conv_3x3', 3), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

# eval-EXP-20211114-021311
PCDARTS_FGSM_es_small = Genotype(
    normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1),
            ('dil_conv_5x5', 3), ('avg_pool_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))  # 0.01

# eval-EXP-20211130-191653
PCDARTS_natural = Genotype(
    normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6),
    reduce=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# eval-EXP-20211201-225407
PCDARTS_caa = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3),
            ('sep_conv_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))

# eval-EXP-20211205-022515
PCDARTS_CAA_1 = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1),
            ('sep_conv_5x5', 2), ('max_pool_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# eval-EXP-20211205-153001
PCDARTS_CAA_es = Genotype(
    normal=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1),
            ('dil_conv_5x5', 3), ('sep_conv_3x3', 4), ('avg_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 2), ('max_pool_3x3', 4), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
PCDARTS = PCDARTS_CAA_es

# eval-CAA_noise_FGSM-20211207-005526
PCDARTS_CAA_noise_FGSM = Genotype(
    normal=[('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2),
            ('max_pool_3x3', 3), ('max_pool_3x3', 3), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# PCDARTS = PCDARTS_Jacobian
# PCDARTS = DARTS_V1
# eval-CAA_noise_PGD-20211207-081059
PCDARTS_CAA_noise_PGD = genotype = Genotype(
    normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 3),
            ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

# eval-DARTS_Arch0-20211208-022638
DARTS_Arch0 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0),
            ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))

PCDARTS_Diff_PGD = Genotype(
    normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0),
            ('sep_conv_5x5', 1), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2),
            ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# eval-DARTS_FGSM-0.1-20211214-003903
DARTS_FGSM = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# eval-NASP_Jacobian-0.1-20211215-063512
NASP_Jacobian = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

# eval-PCDARTS_Jabocian-0.1-20211109-024423
PCDARTS_Jacobian = Genotype(
    normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3),
            ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 3),
            ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

PCDARTS_FGSM_1_10 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 3), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
PCDARTS_PGD_noise = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 3), ('dil_conv_5x5', 3), ('skip_connect', 1)], reduce_concat=range(2, 6))
DARTS_Jacobian = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))
DARTS_FGSM_1_100 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

PCDARTS_System = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('skip_connect', 3), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

NASP_Clean = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))