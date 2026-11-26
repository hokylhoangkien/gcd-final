"""Dataset dispatch outside the immutable upstream snapshot."""


def get_class_splits(args):
    if args.dataset_name == 'rp2k':
        from gcd_runtime.rp2k import class_splits
        return class_splits(args)
    from data.get_datasets import get_class_splits as upstream
    return upstream(args)


def get_datasets(name, train_transform, test_transform, args):
    if name == 'rp2k':
        from gcd_runtime.rp2k import datasets
        return datasets(train_transform, test_transform, args)
    from data.get_datasets import get_datasets as upstream
    return upstream(name, train_transform, test_transform, args)
