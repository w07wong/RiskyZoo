import setuptools

setuptools.setup(
    name='riskyzoo',
    version='0.0.1',
    description='A library for risk sensitive learning.',
    author='William Wong',
    url='https://github.com/w07wong/RiskyZoo',
    packages=setuptools.find_packages(),
    install_requires=['numpy',
                      'torch',
                      'matplotlib'
                      ],
)