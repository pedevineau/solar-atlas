#! /usr/bin/env python

import numpy

#join arrays of same length and remove NaNs and dataerror values
# values = dataerror and values
def remove_nodata(measur,estim,dataerror=None):
	if (len(measur) < 1) or (len(estim) < 1):
		#raise Exception, "Empty data" 
		return None
	if len(measur) != len(estim):
		#raise Exception, "Input data differ in size" 
		return None
	aData=numpy.empty((len(measur),2), dtype='float64')
	aData[:,0]=measur
	aData[:,1]=estim
	acondition=numpy.logical_not(numpy.isnan(aData[:,0])) & numpy.logical_not(numpy.isnan(aData[:,1])) 
	aData2=aData[numpy.where(acondition)]
	if not(dataerror is None):
		acondition=(aData2[:,0] != dataerror) & (aData2[:,1] != dataerror)
		aData2=aData2[numpy.where(acondition)]
	
	if not len(aData2) :
		return None
	return (aData2)

#calculate MBD 
def mbd(measur,estim,percent=False,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	# now calculate the MBE
	if percent:
		return 100*numpy.mean(aData2[:,1]-aData2[:,0])/numpy.mean(aData2[:,0])
	else:
		return numpy.mean(aData2[:,1]-aData2[:,0])

#calculate MAD (MAE) 
def mad(measur,estim,percent=False,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	# now calculate the MAE
	if percent:
		return 100*numpy.mean(abs(aData2[:,1]-aData2[:,0]))/numpy.mean(aData2[:,0])
	else:
		return numpy.mean(abs(aData2[:,1]-aData2[:,0]))

#calculate STDERR 
def stderr(measur,estim,percent=False,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	# now calculate the STDERR
	e_mean=numpy.mean((aData2[:,1]-aData2[:,0]))
	e=(aData2[:,1]-aData2[:,0])
	if percent:
		return 100*numpy.sqrt(numpy.mean((e-e_mean)**2))/numpy.mean(aData2[:,0])
	else:
		return numpy.sqrt(numpy.mean((e-e_mean)**2))

#calculate STDBIAS
def stdbias(measur,estim,percent=False,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	# now calculate the STDBIAS
	std1=numpy.std(aData2[:,1])
	std0=numpy.std(aData2[:,0])
	if percent:
		return 100*(std0-std1)/numpy.mean(aData2[:,0])
	else:
		return (std0-std1)

#calculate DISP
def disp(measur,estim,percent=False,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	# now calculate the DISP
	std1=numpy.std(aData2[:,1])
	std0=numpy.std(aData2[:,0])
	cc=numpy.corrcoef([aData2[:,1],aData2[:,0]])[0,1]
	if percent:
		return 100*(numpy.sqrt(2*std0*std1*(1.-cc)))/numpy.mean(aData2[:,0])
	else:
		return (numpy.sqrt(2*std0*std1*(1.-cc)))


#calculate RMSD 
def rmsd(measur,estim,percent=False,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	
	#now calculate the RMSE	
	if percent:
		return 100*numpy.sqrt(numpy.mean((aData2[:,1]-aData2[:,0])**2))/numpy.mean(aData2[:,0])
	else:
		return numpy.sqrt(numpy.mean((aData2[:,1]-aData2[:,0])**2))

#calculate correlation coeficient
def corr_coef(measur,estim,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	
	#now calculate the RMSE	
	return numpy.corrcoef([aData2[:,1],aData2[:,0]])[0,1]

#calculate average of measured values 
def avg_measured(measur,estim,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	
	#now calculate the avg
	return numpy.mean(aData2[:,0])

#calculate number of valid values 
def valid_pairs_count(measur,estim,dataerror=None):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	
	#now calculate the count
	return len(aData2[:,0])

#print first order measures
def print_err_stat(measur,estim,dataerror=None):
	print "rmbe", mbd(measur,estim,percent=True,dataerror=None)
	print "rmae", mad(measur,estim,percent=True,dataerror=None)
	print "rstderr", stderr(measur,estim,percent=True,dataerror=None)
	print "rstdbias", stdbias(measur,estim,percent=True,dataerror=None)
	print "rdisp", disp(measur,estim,percent=True,dataerror=None)
	print "rrmse", rmsd(measur,estim,percent=True,dataerror=None)
	print "mbe", mbd(measur,estim,percent=False,dataerror=None)
	print "mae", mad(measur,estim,percent=False,dataerror=None)
	print "stderr", stderr(measur,estim,percent=False,dataerror=None)
	print "stdbias", stdbias(measur,estim,percent=False,dataerror=None)
	print "disp", disp(measur,estim,percent=False,dataerror=None)
	print "rmse", rmsd(measur,estim,percent=False,dataerror=None)
	print "cc", corr_coef(measur,estim,dataerror=None)

#print selected first order measures - one line version
def print_err_stat_short(measur,estim,dataerror=None):
	ambd=mbd(measur,estim,percent=False,dataerror=None)
	amad=mad(measur,estim,percent=False,dataerror=None)
	armsd=rmsd(measur,estim,percent=False,dataerror=None)
	acc=corr_coef(measur,estim,dataerror=None)
	armbd=mbd(measur,estim,percent=True,dataerror=None)
	armad=mad(measur,estim,percent=True,dataerror=None)
	arrmsd=rmsd(measur,estim,percent=True,dataerror=None)
	print "#mbd mad rmsd cc rmbd rmad rrmsd"
	print '%.3f %.3f %.3f %.4f %.3f %.3f %.3f ' %(ambd, amad, armsd, acc, armbd, armad, arrmsd)

#calculate cumulative distribution function
def df(aData, (amin,amax), bins=100):
	adf=numpy.histogram(aData, range=(amin,amax),bins=bins,normed=True)[0]
	return adf


#calculate cumulative distribution function
def cdf(aData, (amin,amax), bins=100):
	#acdf=numpy.histogram(aData, range=(amin,amax),bins=bins,normed=True)[0]/bins
	acdf=numpy.histogram(aData, range=(amin,amax),bins=bins,normed=True)[0]
	for i in range(1,bins):
		acdf[i]=acdf[i]+acdf[i-1]
	acdf=acdf/max(acdf) # normalize to 0-1
	return acdf



def create_bin_breaks( nbins, (minim, maxim), doround=False):
	#returns bins breaks in numpy array [binmin,binmax,bbincenter]
	binsize=(maxim-minim)/float(nbins)
	result=numpy.zeros((nbins,3))
	result[:,0]=numpy.linspace(minim,maxim-binsize,nbins) 
	result[:,1]=numpy.linspace(minim+binsize,maxim,nbins)
	result[:,2]=result[:,0]+(binsize/2.)
	if doround: result=result.round()
	return (result)

#calculate KSI - our simplified (operational) version

def ksi(measur,estim,dataerror=None, cdf_min=0, cdf_max=1200, cdf_bin_size=5):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	
	measured=aData2[:,0]
	modelled=aData2[:,1]
	Num=aData2.shape[0]

	# now calculate the KSI
	cdf_bin_num=(cdf_max-cdf_min)/cdf_bin_size

	cdf_ground=cdf(measured, (cdf_min,cdf_max), bins=cdf_bin_num)
	cdf_sat=cdf(modelled, (cdf_min,cdf_max), bins=cdf_bin_num)
	Vc=1.63/numpy.sqrt(Num)
	cdf_diff=numpy.abs(cdf_sat-cdf_ground)

	KSI = 100*cdf_diff.mean()/Vc
	return (KSI)


#calculate KSI - full version
def ksi2(measur,estim,dataerror=None, cdf_min=0, cdf_max=1200, cdf_bin_size=5):
	# removing no value data (can be also value e.g. -999)
	aData2=remove_nodata(measur,estim,dataerror)
	if aData2 is None :
		return None
	
	measured=aData2[:,0] #measured data vector
	modelled=aData2[:,1] #satellite data vector
	Num=aData2.shape[0] #Number of samples
	
	# now calculate the KSI
	cdf_bin_num=(cdf_max-cdf_min)/cdf_bin_size

	cdf_measured=cdf(measured, (cdf_min,cdf_max), bins=cdf_bin_num)
	cdf_modelled=cdf(modelled, (cdf_min,cdf_max), bins=cdf_bin_num)


	cdf_diff_abs=numpy.abs(cdf_modelled-cdf_measured) #absolute difference 
	KSI=numpy.sum(cdf_diff_abs*cdf_bin_size) #trapezoidal integration
	
	Vc=1.63/numpy.sqrt(Num)
	aCritical= Vc*(cdf_max-cdf_min)
	
	KSI_perc = 100*KSI/aCritical
	return (KSI_perc)



if __name__ == "__main__":

	a=numpy.array([0.25,0.1,0.3,0.8,0.6,0.2,0.5,0.7,0.6,0.2,0.1,0.4,0.7,0.9,0.1,0.1,0.2])
	b=numpy.array([0.21,0.21,0.33,0.87,0.69,0.12,0.56,0.67,0.56,0.22,0.11,0.44,0.71,0.92,0.14,0.12,0.23])
	print mbd(a,b)
	print rmsd(a,b)
	print corr_coef(a,b)
	print cdf(a,(0., 1.),8)
	print cdf(b,(0., 1.),8)



