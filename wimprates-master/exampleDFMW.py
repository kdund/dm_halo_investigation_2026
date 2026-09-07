import wimpratesMod as wp
import numpy as np
import matplotlib.pyplot as plt
import numericalunits as nu
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.colors as colors
#All three of below values from fitted MW
#DM density in Msun/kpc^3
DMdens=12e6
#escape velocity from galaxy at sun in km/s
vesc=542
#circulat speed at sun's radius in km/s
v0=237

#Here we use to halo model with the same velocity dispersion but different density
#This merely multiplies all cross sections by a factor, equivalent to changing coupling strength.
halo_modelold = wp.StandardHaloModel(rho_dm=DMdens*nu.Msolar/nu.kpc**3)
etrue_centers = np.geomspace(0.1,1000,400)
mw = 1000
t=59.37
#see example_dfnew.py to see how to write files like this
halo_modelnew=wp.HaloModelInterpolatedFromFile(Filename="wimpratesMod/data/dataPot/fourCoef.txt",
                                               rho_dm=DMdens*nu.Msolar/nu.kpc**3)
t=59.37
for color, target in zip(["crimson","darkorange","darkgreen","cyan"],["Xe","Ge","Ar","Si"]):
    r_new=wp.rate_wimp_std(etrue_centers,mw=mw,sigma_nucleon=1e-45, halo_model=halo_modelnew, material=target)
    r_old=wp.rate_wimp_std(etrue_centers,mw=mw,sigma_nucleon=1e-45, halo_model=halo_modelold, material=target)
    plt.plot(etrue_centers, r_new,color=color,linestyle="-", label=target+" from fitted DF")
    plt.plot(etrue_centers, r_old,color=color,linestyle="--", label=target+" from Gaussian DF")
    cs=CubicSpline(etrue_centers,r_new)
    r1=quad(cs,min(etrue_centers),max(etrue_centers))[0]
    cs2=CubicSpline(etrue_centers,r_old)
    r2=quad(cs2,min(etrue_centers),max(etrue_centers))[0]
    print("rate:",r1,r2)
plt.yscale("log")
plt.xlim(0.7,3000)
plt.yscale("log")
plt.ylim(1e-6,100)
plt.xscale("log")
plt.legend()
plt.xlabel("Recoil energy [$\mathrm{keV}$]")
plt.ylabel("Recoil rate [$\mathrm{t}^{-1}\mathrm{y}^{-1}\mathrm{keV}^{-1}$]")
plt.title("SI WIMP-nucleon recoil rate, $\sigma=10^{-45}\mathrm{cm}^{-2}$, M=$500\mathrm{GeV}/c^2$")
plt.savefig("wimprates_targets2.png")
plt.show()
v=np.linspace(0.1,800,300)
A=halo_modelnew.velocity_dist(v*nu.km/nu.s,t)*nu.km/nu.s
plt.plot(v, halo_modelold.velocity_dist(v*nu.km/nu.s,t)*nu.km/nu.s,color="blue",label="Standard halo model")
plt.plot(v,A, label="df halo",color="red")
plt.legend()
plt.xlabel("v (km/s)")
plt.xlim(0,800)
plt.ylabel("Probability (km/s)^-1")
plt.ylim(0,0.003)
plt.show()
