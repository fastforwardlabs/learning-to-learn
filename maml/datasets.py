import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet, QuickDraw
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose

from maml.model import ModelConvQuickDraw
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')

def get_benchmark_by_name(name,
                          folder,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None,
                          random_seed=123,
                          num_training_samples=100):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'quickdraw':
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = QuickDraw(folder,
                                       transform=transform,
                                       target_transform=Categorical(num_ways),
                                       num_classes_per_task=num_ways,
                                       meta_train=True,
                                       dataset_transform=dataset_transform,
                                       download=True,
                                       random_seed=random_seed,
                                       num_training_samples = num_training_samples)
        meta_val_dataset = QuickDraw(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_val=True,
                                     dataset_transform=dataset_transform)
        meta_test_dataset = QuickDraw(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_test=True,
                                      dataset_transform=dataset_transform)

        model = ModelConvQuickDraw(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)
