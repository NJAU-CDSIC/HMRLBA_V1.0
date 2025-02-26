import os
import argparse
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

from hmrlba_code.data.handlers import PDBBind, Enzyme

HANDLERS = {'pdbbind': PDBBind, 'enzyme': Enzyme}
DATA_DIR = os.path.join(os.environ['PROT'], "Datasets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data Directory")
    parser.add_argument("--prot_mode", default='surface2backbone')
    parser.add_argument("--dataset", default='pdbbind', help="Dataset for which we prepare graphs")
    parser.add_argument("--pdb_ids", nargs="+", default=None)
    parser.add_argument("--plm", default='esm1b', help="PLM to use")
    args = parser.parse_args()

    handler_cls = HANDLERS.get(args.dataset)

    kwargs = {}

    handler = handler_cls(dataset=args.dataset,
                          data_dir=args.data_dir,
                          prot_mode=args.prot_mode,
                          plm=args.plm, **kwargs)
    handler.process_ids(pdb_ids=args.pdb_ids)


if __name__ == "__main__":
    main()
