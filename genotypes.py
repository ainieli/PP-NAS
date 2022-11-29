from collections import namedtuple

from hanser.models.cifar.ppnas.genotypes import PP_ResNet_CIFAR10_FAIR_AVD_2, PP_ResNet_CIFAR100_FAIR_AVD_1, PP_ResNet_ImageNet_FAIR_AVD_1


Genotype = namedtuple('Genotype', 'normal')

CIFAR10_Geno = PP_ResNet_CIFAR10_FAIR_AVD_2
CIFAR100_Geno = PP_ResNet_CIFAR100_FAIR_AVD_1
ImageNet_Geno = PP_ResNet_ImageNet_FAIR_AVD_1





