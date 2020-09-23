import torch
import math
import os
import time
import json
import logging
import numpy as np

from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if isinstance(v, float):
            v = "{:.{}f}".format(v, 5) 
        if isinstance(v, int):
            v = str(v)
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))  

def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')
    
    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))

        folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        outfile_path = os.path.abspath(os.path.join(folder, 'model_results.json'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))

    benchmark = get_benchmark_by_name(args.dataset,
                                      args.folder,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      hidden_size = args.hidden_size,
                                      random_seed = args.random_seed,
                                      num_training_samples = args.num_training_samples)

    meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=benchmark.loss_function,
                                            device=device)
    #print(benchmark.model)
    
    best_value = None
    output = []
    pretty_print('epoch', 'train loss', 'train acc', 'train prec', 'val loss', 'val acc', 'val prec')
    
    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    for epoch in range(args.num_epochs):
        train_results = metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False)
        val_results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        pretty_print((epoch+1), train_results['mean_outer_loss'], 
                     train_results['accuracies_after'], 
                     train_results['precision_after'], 
                     val_results['mean_outer_loss'], 
                     val_results['accuracies_after'],
                     val_results['precision_after'])
        
        # Save best model
        if 'accuracies_after' in val_results:
            if (best_value is None) or (best_value < val_results['accuracies_after']):
                best_value = val_results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > val_results['mean_outer_loss']):
            best_value = val_results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)
        # saving results for later use - plotting, etc.       
        output.append({
            'epoch': (epoch+1),
            'train_loss': train_results['mean_outer_loss'],
            'train_acc': train_results['accuracies_after'], 
            'train_prec': train_results['precision_after'], 
            'val_loss': val_results['mean_outer_loss'],
            'val_acc': val_results['accuracies_after'],
            'val_prec': val_results['precision_after']
        })
        if (args.output_folder is not None):
            with open(outfile_path, 'w') as f:
                json.dump(output, f)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['quickdraw'], default='quickdraw',
        help='Name of the dataset (default: quickdraw).')
    
    parser.add_argument('--num-training-samples', type=int, default=100,
        help='Number of training examples per class for down-sampling (default: 100).')
    parser.add_argument('--random-seed', type=int, default=123,
        help='Random seed to split Quickdraw classes across train-val-test (default: 123).')
    parser.add_argument('--output-folder', type=str, default='./models',
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=1,
        help='Number of training example per class (k in "k-shot", default: 1).')
    parser.add_argument('--num-shots-test', type=int, default=1,
        help='Number of test examples per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 1).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=20,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.01,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--use-cuda', action='store_true', default=True)

    args = parser.parse_args()
    print(torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu'))

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots
    print(args)
    main(args)