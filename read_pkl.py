import pickle
import pathlib
import sys

OUTPUT_DIR = pathlib.Path('./outputs')
FOLDER_PREFIX = 'Covid'
TARGET_SUBPATH = ('scores', 'scores.pkl')

def find_targets():
    targets = []
    if not OUTPUT_DIR.is_dir():
        return targets
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if folder.is_dir() and folder.name.startswith(FOLDER_PREFIX):
            f = folder.joinpath(*TARGET_SUBPATH)
            if f.is_file():
                targets.append(f)
    return targets

def load_and_print(path: pathlib.Path):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"\nLoad {path} Failed: {e}")
        return

    print(f"\nLoad {path}")
    print(f"Type: {type(obj)}")
    print("Content:")
    try:
        print(obj)
    except Exception as e:
        print(f"Error while Printing: {e}")

def list_targets(targets):
    print(f"\nFound {len(targets)} files:\n")
    for i, t in enumerate(targets, 1):
        print(f"  [{i}] {t}")

def main():
    targets = find_targets()

    if not OUTPUT_DIR.is_dir():
        print(f"Error Directory not found: {OUTPUT_DIR.resolve()}")
        return
    if not targets:
        print(f"Error in {OUTPUT_DIR.resolve()} no files found "
              f"with prefix '{FOLDER_PREFIX}' and filename '{TARGET_SUBPATH[-1]}'")
        return

    list_targets(targets)

    while True:
        try:
            user_input = input(
                f"\nSelect (1-{len(targets)})，"
                f"Input r Rescan, q Exit:"
            ).strip()
        except EOFError:
            print("\nExit")
            break

        if user_input.lower() in ('q', 'quit', 'exit'):
            print("Exit")
            break

        if user_input.lower() == 'r':
            targets = find_targets()
            if not targets:
                print("No files found, please check the directory and try again.")
                break
            list_targets(targets)
            continue

        if not user_input.isdigit():
            print(f"Please enter a number between 1-{len(targets)}, r or q")
            continue

        idx = int(user_input)
        if not (1 <= idx <= len(targets)):
            print(f"Number out of range, should be 1-{len(targets)}")
            continue

        load_and_print(targets[idx - 1])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExit")
        sys.exit(0)