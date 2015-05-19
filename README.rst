=================================================
Diastema: Musical intervals and modality analysis
=================================================

Overview
========

**Diastema**, is a project to analyse musical modality. It uses Python2.

For now, it's main features are :

* Fondamental frequencies extraction (using *PredominentMelody()* from **Essentia**);
* Getting the main frequencies as peaks of the probability density function from frequencies;
* Comparing PDFs using a correlation coefficient;
* Getting a similarity matrix between melodies.

Installation
============

First, you need to install manually **Essentia** (http://essentia.upf.edu/).

Then, install Diastema with the following :

.. code:: python

	git clone https://github.com/AnasGhrab/Diastema
	python setup.py install


Usage
=====

To use Diastema :

.. code:: python

	from distema import *
	path = "path/to/a/folder/with/audios/wav/files/"
	Music = Melodies(path)
	
Then you can

.. code:: python

	Music.PdfsPlot()
	Music.Simatrix()
		
Contact
=======

Homepage: http://anas.ghrab.tn

Email:

 * Anas Ghrab <anas.ghrab@gmail.com>

License
=======

CeCILL v2, compatible with GPL v2 (see `LICENSE <http://github.com/yomguy/Telemeta/blob/master/LICENSE.txt>`_)
