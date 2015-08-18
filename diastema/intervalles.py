"""
.. module:: intervalles
   :platform: Unix, Windows
   :synopsis: Core operations on intervals.

.. moduleauthor:: Anas Ghrab <anas.ghrab@gmail.com>


"""

from numpy import log2,log10

inter = {'2/1*4/3':2/1.*4/3.,'2/1*5/4':2/1.*5/4.,'2/1*6/5':2/1.*6/5.,
         '2/1*9/8':2/1.*9/8.,'2/1*10/9':2/1.*10/9.,'2/1*12/11':2/1.*12/11.,
         '2/1':2/1.,'3/2*5/4':3/2.*5/4.,'3/2*6/5':3/2.*6/5.,'3/2*9/8':3/2.*9/8.,
         '3/2*10/9':3/2.*10/9.,'3/2':3/2.,'4/3':4/3.,'5/4':5/4.,'9/8*12/11':9/8.*12/11.,
          '6/5':6/5.,'9/8':9/8.,'10/9':10/9.,'12/11':12/11.,'1/1':1/1.}
Int = [] # A list where the method convert_ref() puts converted ratio into a linear space
unit = "savart" # Sets the default unit to be savart
        
def set_unit(x):
    """Sets the unit to be used : possible units are savart and cent."""
    global unit
    unit_types = ["savart","cent"]
    if x in unit_types:
        unit = x
    else:
        raise ValueError("Accepted units : 'savart','cent'")

def get_unit():
    """Gets the currently used unit."""
    global unit
    print unit

def cent(y):
    """Converts the given interval (as ratio) to cent."""
    inter_cent = log2(y)*1200
    return inter_cent

def savart(y):
    """Converts the given interval (as ratio) to savart."""
    inter_savart = log10(y)*1000
    return inter_savart

def dias(y,unit="savart"):
    """Converts the given interval (as ratio) to the global chosen unit. Default is savart."""
    if unit=="savart":
        y = savart(y)
    if unit == "cent":
        y = cent(y)
    return y

def convert_ref():
    """Converts all the reference interval set by the global variable inter from ratios to a linear space."""
    global inter, Int
    I = inter.values()
    for i in range(0,len(I)):
        Int.append(dias(I[i])) # Tous les intervalles en savarts
    return Int

def get_inter_ref(x):
    """
        Look for the closest epimoric interval

        Input :
        -----------
            x : an interval in Savart/Cent

        Output :
        -----------
            y : the closest interval from The Interval List Int()
    """
    global inter, Int
    dist = []
    for j in range(0,len(Int)):
        dist.append(abs(abs(x)-abs(Int[j])))
    Index_Intervalle_Proche = dist.index(min(dist))
    Nom_Intervalle_Proche = inter.keys()[dist.index(min(dist))]
    Difference = "{:.2f}".format(min(dist))
    if x>=log10(inter.values()[dist.index(min(dist))])*1000:
        signe = '+'
    else: signe  = '-'
    y = "{:.2f}".format(x),Nom_Intervalle_Proche,signe, Difference
    return y

def get_echelle(echelle):
    for i in range(0,len(echelle)):
        print get_inter_ref(echelle[i])
