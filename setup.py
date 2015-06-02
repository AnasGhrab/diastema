from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='diastema',
      version='0.0.2',
      description='A tool for modal music analysis : Scale analysis, tonic detection',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Musicology :: Analysis',
      ],
      keywords='musicology analysis scale pitch makam melody',
      url='https://github.com/AnasGhrab/diastema',
      author='Anas Ghrab',
      author_email='anas.ghrab@gmail.com',
      license='MIT',
      packages=['diastema'],
      install_requires=[
          'numpy','matplotlib','essentia',
      ],
      zip_safe=False)