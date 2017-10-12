import sys
from pathlib2 import Path

from utils import inspect_checkpoint

if __name__ == '__main__':
    assert len(sys.argv) > 1
    ckpt_path = sys.argv[1]
    if ckpt_path.endswith('tar'):
        inspect_checkpoint(Path(ckpt_path))
    else:
        inspect_checkpoint(Path(ckpt_path) / 'checkpoint.pth.tar')

