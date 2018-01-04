from omnium.version import get_version

VERSION = (0, 5, 0, 0, 'alpha')


__version__ = get_version()

if __name__ == '__main__':
    print(get_version())
