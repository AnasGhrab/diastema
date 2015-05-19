=================================================
Diastema: Musical intervals and modality analysis
=================================================

|version| |downloads| |travis_master| |coverage_master|

.. |travis_master| image:: https://secure.travis-ci.org/Parisson/Telemeta.png?branch=master
   :target: https://travis-ci.org/Parisson/Telemeta/
   :alt: Travis

.. |version| image:: https://pypip.in/version/Telemeta/badge.png
   :target: https://pypi.python.org/pypi/Telemeta/
   :alt: Version

.. |downloads| image:: https://pypip.in/download/Telemeta/badge.svg
   :target: https://pypi.python.org/pypi/Telemeta/
   :alt: Downloads

.. |coverage_master| image:: https://coveralls.io/repos/Parisson/Telemeta/badge.png?branch=master
   :target: https://coveralls.io/r/Parisson/Telemeta?branch=master
   :alt: Coverage

Overview
========

**Diastema**, is a project to analyse musical modality. It uses Python2.

For now, it's main features are :

* Fondamental frequencies extraction (using _PredominentMelody()_ from **Essentia**);
* Getting the main frequencies as peaks of the probability density function from frequencies;
* Comparing PDFs using a correlation coefficient;
* Getting a similarity matrix between melodies.

Installation
============

To install Diastema::
	git clone https://github.com/AnasGhrab/Diastema
	python setup.py install

You need to install manually Essentia (http://essentia.upf.edu/)

Contact
=======

Homepage: http://anas.ghrab.tn

Email:

 * Anas Ghrab <anas.ghrab@gmail.com>

License
=======

CeCILL v2, compatible with GPL v2 (see `LICENSE <http://github.com/yomguy/Telemeta/blob/master/LICENSE.txt>`_)
