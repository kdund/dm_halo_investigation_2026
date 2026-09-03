import wimpratesMod as wp
import numpy as np
import matplotlib.pyplot as plt
import numericalunits as nu

import agama
def createNewDoublePowerLawDF(**params):
    def df(J):
        if(np.sum(J)==0):
            if(np.shape(J)[0]==1):
                return np.nan_to_num(norm / (2*np.pi * J0)**3*(J0/Jcore)**slopeIn)
            else:
                return np.ones(np.shape(J)[0])*np.nan_to_num(norm / (2*np.pi * J0)**3*(J0/Jcore)**slopeIn)
        modJphi = abs(J[:,2])
        L = J[:,1] + modJphi
        c = L / (L + J[:,0])
        jt = (1.5 * J[:,0] + L) / L0
        jta = jt**alpha
        xi = jta / (1+jta)
        rat = (1-xi) * Fin + xi * Fout
        a = 0.5 * (rat+1)
        b = 0.5 * (rat-1)
        cL = np.where(L>0, J[:,1] * (a + b * modJphi / L) + modJphi, 0)
        fac = np.exp(beta * np.sin(np.pi/2 * c))
        hJ = J[:,0] / fac + 0.5 * (1 + c * xi) * fac * cL
        gJ = hJ
        result = norm / (2*np.pi * J0)**3 * (1 + J0/hJ)**slopeIn * (1 + gJ/J0)**(-slopeOut)
        if Jcutoff > 0:
            result *= np.exp(-(gJ / Jcutoff)**cutoffStrength)
        if Jcore > 0:
            result *= (1 + Jcore/hJ * (Jcore/hJ - zeta))**(-0.5*slopeIn)
        if rotFrac != 0:
            result *= 1 + rotFrac * np.tanh(J[:,2] / Jphi0)
        return result
    J0, L0, slopeIn, slopeOut, rotFrac, Jphi0, alpha, beta, Fin, Fout, Jcutoff, cutoffStrength, Jcore = (
        float(params[name]) for name in 
        ('J0', 'L0', 'slopeIn', 'slopeOut', 'rotFrac', 'Jphi0', 'alpha', 'beta',
        'Fin', 'Fout', 'Jcutoff', 'cutoffStrength', 'Jcore'))
    if Jcore > 0:
        import scipy.optimize, scipy.integrate
        def rootFnc(zeta):
            def integrand(t):
                hJ = Jcore * t*t*(3-2*t) / (1-t)**2 / (1+2*t)
                dhJdt = Jcore * 6*t / (1-t)**3 / (1+2*t)**2
                return (hJ**2 * dhJdt * (1 + J0/hJ)**slopeIn * (1+hJ/J0)**(-slopeOut) *
                    ((1 + Jcore/hJ * (Jcore/hJ-zeta))**(-0.5*slopeIn) - 1))
            return scipy.integrate.fixed_quad(integrand, 0.0, 1.0, n=20)[0]
        zeta = scipy.optimize.brentq(rootFnc, 0.0, 2.0)
    norm = 1.0
    norm = float(params['mass']) / agama.DistributionFunction(df).totalMass()
    return df
import os,sys

try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3
iniFileName = "dataPot/SCM_MW.ini"
ini = RawConfigParser()
ini.optionxform=str  # do not convert key to lowercase
ini.read(iniFileName)
iniDFDarkHalo    = dict(ini.items("DF dark halo"))
Rsol      = ini.getfloat("Data", "SolarRadius")
dfDarkHalo    = createNewDoublePowerLawDF(**iniDFDarkHalo)

#height of sun
zsol=0.025
#total Potential
potTot=agama.Potential("dataPot/mwmodel_potential.ini")
#action finder
af=agama.ActionFinder(potTot)
#DM density- does not affect rate calculation in Msun/kpc^3
DMdens=12e6
def dfDM(velssc):
    vels=velssc*nu.s/nu.km
    if np.sum(vels**2)**0.5>=vesc:
        return 0
    J=af(np.array([Rsol,0,zsol,vels[0],vels[1],vels[2]]))
    if(np.isnan(J[0]+J[1])):
        return 0
    DistF=dfDarkHalo(np.array([J]))[0]/DMdens
    if np.isnan(DistF):
        print("nan:",np.sum(vels**2)**0.5/vesc,J,dfDarkHalo(np.array([J])))
    return DistF*1/(nu.km/nu.s)**3
#escape velocity from galaxy at sun
vesc=np.sqrt(-2*potTot.potential([Rsol,0,zsol]))
#circulat speed at sun's radius
v0=np.sqrt(Rsol*(-potTot.force([Rsol,0,0])[0]))


halo_new = wp.StandardHaloModel(
    v_0=v0*nu.km/nu.s,
    v_esc=vesc*nu.km/nu.s,
    )
halo_modelnew=wp.HaloModelInterpolatedt(dfDM,v_0=v0*nu.km/nu.s,v_esc=vesc*nu.km/nu.s,N=100)
v=np.linspace(0.1,800,300)
t=59.67
A=halo_modelnew.velocity_dist(v*nu.km/nu.s,t)*nu.km/nu.s
print("no:",A)
plt.plot(v,halo_new.velocity_dist(v*nu.km/nu.s,t)*nu.km/nu.s,color="blue",label="Standard halo model")
plt.plot(v,A, label="df halo",color="red")
plt.legend()
plt.xlabel("v (km/s)")
plt.xlim(0,800)
plt.ylabel("Probability (km/s)^-1")
plt.ylim(0,0.003)
plt.show()
