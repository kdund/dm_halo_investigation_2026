from xe_likelihood import BinwiseInference, Spectrum

import scipy.stats as sps
import numpy as np
from tqdm.notebook import trange,tqdm
import matplotlib.pyplot as plt
import time
import matplotlib.colors as colors
from scipy.interpolate import interp1d,CubicSpline
import wimprates as wp
import numericalunits as nu
etrue_centers = np.linspace(0.01,100,101)
xenon_inference = BinwiseInference.from_xenonnt_mc(spectrum = sps.norm(10, 0.1).pdf)
#change path to location of wimprates-master fork
#Herevthe halo model has the same density as the standard halo model.
halomodel=wp.HaloModelInterpolatedFromFile(
    "wimprates-master-2/wimprates/data/dataMW1/FourCoef.txt")
halo2=wp.StandardHaloModel()
def nonsub_llr(cinference, spectrum=None, normalised=True, run=0):
    if spectrum is not None:
        cinference.spectrum = spectrum
        

    srms    = np.linspace(0, cinference.srm_max, 1000)
    llrs    = np.zeros(len(srms))
    likelihood_matrix = cinference.likelihood_matrix[run]
    for i, mu_bin in enumerate(cinference.spectrum_in_reconstructed_bins):
        f_llr_bin = interp1d(cinference.signal_expectations_bins, likelihood_matrix[i, :],
                                 bounds_error=False, fill_value=np.inf)
        llrs += f_llr_bin(mu_bin * srms)
    if normalised:
        llrs -= llrs.min()
    f_llr = interp1d(srms, llrs, bounds_error=False, fill_value=np.inf)
    return f_llr

srms = np.geomspace(1e-47,.5e-45,100)/1e-45
massdp=np.geomspace(10,1e3,100)
recoilspec=[]
llrs = np.zeros((len(massdp), len(srms)))
for i in range(len(massdp)):
    rec=wp.rate_wimp_std(etrue_centers, mw=massdp[i], sigma_nucleon=1e-45, halo_model=halo2)
    s =Spectrum.from_sample(etrue_centers,rec)
    xenon_inference.spectrum=s

    llr = nonsub_llr(xenon_inference, spectrum=s, normalised=False)
    llrs[i] = llr(srms)

print(llrs.min(),llrs.max())
llrs -= np.min(llrs)
plt.clf()
I=plt.pcolormesh(massdp, srms*1e-45, llrs.T,cmap="gray",vmax=10)
plt.contour(massdp, srms*1e-45, llrs.T,colors="magenta",levels=[-np.log(0.9),-np.log(0.1),-np.log(0.01)])
plt.colorbar(I)
plt.xlabel("Mw ($Gev/c^2$)")
plt.ylabel("$\\sigma_0 (cm^{-2})$")
plt.xlim(min(massdp),max(massdp))
plt.ylim(min(srms*1e-45),max(srms)*1e-45)
plt.yscale("log")
plt.xscale("log")
plt.show()
