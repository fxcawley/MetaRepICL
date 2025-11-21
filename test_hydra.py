import argparse
import hydra

# Patch argparse to handle Hydra's LazyCompletionHelp on Python 3.14
_orig_add_argument = argparse.ArgumentParser.add_argument

def _safe_add_argument(self, *args, **kwargs):
    if 'help' in kwargs:
        h = kwargs['help']
        if h.__class__.__name__ == 'LazyCompletionHelp':
            # Python 3.14 requires help to be a string (checks for '%')
            kwargs['help'] = "Shell completion (patched)"
    return _orig_add_argument(self, *args, **kwargs)

argparse.ArgumentParser.add_argument = _safe_add_argument

@hydra.main(version_base=None)
def main(cfg):
    print("Hydra works with argparse patch")

if __name__ == "__main__":
    main()