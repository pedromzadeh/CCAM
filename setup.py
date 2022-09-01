from setuptools import setup
setup(
    name='collider',
    version='1.0',
    author='Pedrom Zadeh',
    description='2D cell collisions simulated within phase-field framework',
    packages=['src', 'visuals'],
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'pandas',
        'matplotlib',
        'scikit-image',
        'scikit-learn',
        'statsmodels'   
    ])