from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='femida-detect',
        author='Brazhenko Dmitry, Maxim Kochurov',
        packages=find_packages(),
        install_requires=open('requirements.txt', 'r').readlines(),
        tests_require=open('requirements-dev.txt', 'r').readlines()
    )
