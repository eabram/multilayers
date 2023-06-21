from setuptools import setup

setup(name='multilayers',
      version='1.0.1',
      description='Multilayer calculation with the Tranfer Matrix Method also including absoption profiles',
      url='https://git.amolf.nl/abram/multilayers',
      author='Ester Abram',
      author_email='E.abram@arcnl.nl',
      license='ARCNL',
      packages=['multilayers'],
      zip_safe=False,
      install_requires=['numpy','scipy','pandas','matplotlib','os','sys'])
