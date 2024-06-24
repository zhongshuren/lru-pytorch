import torch
from args import parser
from dataloading import Datasets
from lru import StackedLRU
from train import train


def main(args):
    create_dataset_fn = Datasets[args.dataset]
    # Dataset dependent logic
    if args.dataset == "copy-classification":
        assert args.pooling == "none", "No pooling for copy task"
        dense_targets = True
    else:
        dense_targets = False
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        if args.dataset in ["aan-classification"]:
            # Use retrieval model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False
    else:
        retrieval = False

    (
        trainloader, valloader, testloader, aux_dataloaders,
        n_classes, seq_len, in_dim, train_size,
    ) = create_dataset_fn(args.dir_name, seed=1007, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedLRU(args, input_dim=in_dim, out_dim=n_classes, multidim=2)
    model = model.to(device)

    train(args, model, (trainloader, valloader, testloader), device)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
