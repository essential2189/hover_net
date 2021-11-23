import os
import sys

def main(path):
    print(path)
    if os.path.exists(path):
        print("File exist")

if __name__ == '__main__':
    main(sys.argv[1])