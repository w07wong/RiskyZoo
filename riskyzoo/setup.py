import setuptools

setuptools.setup(
    name='riskyzoo',
    version='0.0.1',
    description='A library for risk sensitive learning.',
    author='William Wong',
    url='',
    packages=setuptools.find_packages(),
    install_requires=['numpy',
                      'torch',
                      'matplotlib'
                      ],
)