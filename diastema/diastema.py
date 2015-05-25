import numpy
import matplotlib.pyplot as plt
#from scipy import stats
#from scipy import signal
from scipy.stats.kde import gaussian_kde
from scipy.stats.mstats import mode

import glob, os.path, time, os

from essentia import *  # Importer toutes les fonctions de la bibliotheque Essentia
from essentia.standard import * # Importer toutes les fonctions du module **standard** de Essentia
#from pandas import *

class Melodie(object):
	"""Une classe definissant une melodie et ses caracteristiques, a partir de la liste de ses frequences : 
	- les frequences minimales et maximales
	- la frequence moyenne et ecart-type
	- la frequence dominante
	- la fonction de probabilite de ses frequences

	"""
		
	def __init__(self, file):
		self.file_path = file
		self.file_name = os.path.split(self.file_path)[1]
		self.file_label = self.file_name.split(".")[0]
		self.file_exten = self.file_name.split(".")[1]
		self.freqref = 500

		if self.file_exten == 'txt':
			self.frequences = numpy.loadtxt(file)
			self.freq = self.frequences[~numpy.isnan(self.frequences)]
			self.freqtransmode = self.transmode(self.freqref)
			self.pdf = gaussian_kde(self.freqtransmode[~numpy.isnan(self.freqtransmode)],bw_method=.05)
			self.fmin = min(self.freq)
			self.fmax = max(self.freq)
			self.fmean = numpy.mean(self.freq)
			self.fstd = numpy.std(self.freq)
			self.x = numpy.linspace(300,700,1000)
			self.pdf = self.pdf(self.x)
			self.peaks()
		if self.file_exten == 'wav':
			self.file_pitch_extract(self.file_path)

	def __str__(self):
		return "File : %s" % (self.file_name)

	def file_pitch_extract(self,file):
		"""Extraction des frequences avec PredominantMelody()
		Le resultat est un fichier .txt"""

		start = time.time()
		print start,' : Extraction des f0 de ',self.file_name

		audio = MonoLoader(filename = file)() # creation de l'instance
		melodie = PredominantMelody() # creation de l'instance
		pitch, confidence = melodie(audio) 
		pitch[pitch==0]=numpy.nan
		nom_fichier = self.file_name.replace("wav", "txt");
		numpy.savetxt(os.path.split(self.file_path)[0]+'/txt/'+nom_fichier,pitch,fmt='%1.3f')

		end = time.time()
		print 'Fichier',self.file_label,'analyse (',end - start,') secondes'

		return
	
	def pdf(self, x):
		"""
		Estime la densite de probabilite

		Parameters
		----------
		x : array_like

		Returns
		-------
		y : array_like
	    	Returns a ....
	    """

		return self.pdf.evaluate(x)

	def pdf_show(self):
	 	"""Affichage de la fonction de la densite de probabilite"""

		plt.plot(self.x,self.pdf,linewidth=3,alpha=.6,label=self.file_label)
		plt.plot(self.peaks, self.peakspdf, "ro")
		for i in range(0,len(self.peaks)):
			plt.annotate(
					"%.2f" % self.peaks[i],
					xy=(self.peaks[i], self.peakspdf[i]),
					xytext=(self.peaks[i], (self.peakspdf[i]+0.0001)))
		plt.legend()
		#return plt.figure()

	def peaks(self):
		"""Obtenir les Peaks du PDF

		"""

		c = (numpy.diff(numpy.sign(numpy.diff(self.pdf))) < 0).nonzero()[0] + 1 # local max

		self.peaks = self.x[c]
		self.peakspdf = self.pdf[c]

		#
		### EN UTILISANT signal.find_peaks_cwt
		# self.frequences = self.frequences[~numpy.isnan(self.frequences)] 
		# X = numpy.linspace( min(self.frequences), max(self.frequences), 100 )
		# PDF = self.pdf(X)
		# widths = numpy.arange(1,20)
		# self.peakind = signal.find_peaks_cwt(PDF, widths)
		# self.peaks = X[self.peakind]
		# self.peakspdf = self.pdf(X)[self.peakind]

		return self.peaks, self.peakspdf

	def transmode(self,freqref):
		"""Transpose all the frequencies by setting the mode on a given reference frequency

		"""
		interv_transpo = mode(self.freq)[0]/freqref
		self.freqtransposed = self.freq / interv_transpo
		return self.freqtransposed

	def tonique(self,percent=8):
		"""
		Get the tonic frequency defined as the mode of the last frequencies array.
		These as selected by the percent argument.
		
		Input :
		-----------
			percent : a percentage of the number of frames from the total size
			of the frequencies array to give the last frequencies. Default percent= 8
		
		Output :
		-----------

			M : the mode
			N : the mode converted inside an octave
			Final_Freqs : the last frequencies according to the percentage
		"""

		L = len(self.freq)
		Nb_Frames = L*percent/100
		Final_Freqs = self.freq[(L-Nb_Frames):L]
		M = mode(Final_Freqs)
		if M[0] > mode(self.freq)[0]*2:
			N = M[0]/2
		if M[0] < mode(self.freq)[0]/2:
			N = M[0]*2 
		return M[0],N,Final_Freqs

class Melodies(object):
	"""Une classe definissant un ensemble de melodies, leur degre d'homogeneite et de proximite

	Attributs:
		- path : un dossier contenant les fichiers .wav ou .txt
	"""

	def __init__(self, path):
		self.path = path  # L'adresse obtenu = un dossier
#		self.exten = exten
		try:
		    os.makedirs(self.path+'txt/')
		except OSError:
		    pass

		self.folder_txt = glob.glob(path+'txt/'+'*.txt')  # Tous les fichiers .txt du dossier
		self.folder_wav = glob.glob(path+'*.wav')  # Tous les fichiers .wav du dossier

		self.melodies = []

		if len(self.folder_txt) == len(self.folder_wav):
			print 'Lecture et analyse de ',len(self.folder_wav),' fichiers (.txt) dans le dossier :',self.path
			for melodie in self.folder_txt:
				self.melodies.append(Melodie(melodie))
			#self.Simatrix()
		else:
			print 'Analyse de ',len(self.folder_wav),' fichiers Audio (.wav) dans le dossier :'
			for melodie in self.folder_wav:
				self.melodies.append(Melodie(melodie))
			for melodie in self.folder_txt:
				self.melodies.append(Melodie(melodie))
		
		self.PdfCorr()
		
	def pitch_extract(self):
		"""Extrait les frequences f0 des tous les fichiers .wav du dossier"""

		files = glob.glob(self.path+'*.wav')
		#fichiers = []
		for fichier_audio in self.melodies:
			self.file_pitch_extract(fichier_audio)
		return

	def PdfCorr(self):
		"""Cree la matrice des coefficients de correlation a partir des pdfs, classe sur la premiere colonne

		"""
		PDFS = []
		for i in range(0,len(self.melodies)):
		    PDFS.append(self.melodies[i].pdf)
		self.PdfCorr = numpy.corrcoef(PDFS)
		return self.PdfCorr

	def PdfsPlot(self):
		"""Dessine les PDFs de tous les fichiers

		"""
		plt.figure(figsize=(18, 12))
		for melodie_pdf in self.melodies:
			melodie_pdf.pdf_show()
		return plt.show()

	def Simatrix(self):
		"""Cree la matrice des coefficients de correlation a partir des pdfs

		"""
		R = self.PdfCorr
		l = len(self.PdfCorr)
		plt.pcolor(R)
		plt.colorbar()
		plt.yticks(numpy.arange(0.5,l+0.5),range(1,l+1))
		plt.xticks(numpy.arange(0.5,l+0.5),range(1,l+1))
		return plt.show()

	def SimPdf(self,i):
		"""PDF des similarites, basee sur la melodie i

		"""
		#self.PdfCorr = self.PdfCorr[:, self.PdfCorr[i].argsort()]
		PdfCorrI = self.PdfCorr[i]
		b = gaussian_kde(PdfCorrI)
		x = numpy.arange(0,1,0.001)
		Y = b(x)
		plt.plot(x,Y)

	def SimPdfs(self):
		"""PDFs des similarites, basee sur toutes les SimPdf

		"""
		for i in range(0,len(self.PdfCorr)):
			self.SimPdf(i)