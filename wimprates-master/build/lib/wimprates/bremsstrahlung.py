"""Elastic nuclear recoil detected through Bremsstrahlung

Kouvaris/Pradler [arxiv:1607.01789v2]
"""
import numericalunits as nu
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
try:
    import agama
except ImportError: agama=None
import wimprates as wr
export, __all__ = wr.exporter()


# Load the X-ray form factor
def to_itp(fn):
    x, y = np.loadtxt(wr.data_file('bs/' + fn), delimiter=',').T
    return interp1d(x, y, fill_value='extrapolate')


f1 = to_itp('atomic_form_1')
f2 = to_itp('atomic_form_2')


def vmin_w(w, mw, material):
    """Minimum wimp velocity to emit a Bremsstrahlung photon w

    :param w: Bremsstrahlung photon energy
    :param mw: WIMP mass

    From Kouvaris/Pradler [arxiv:1607.01789v2], equation in text below eq. 10
    """
    return (2 * w / wr.mu_nucleus(mw, material))**0.5


def erec_bound(sign, w, v, mw, material):
    """Bremsstrahlung scattering recoil energy kinematic limits
    From Kouvaris/Pradler [arxiv:1607.01789v2], eq. between 8 and 9,
    simplified by vmin (see above)

    :param sign: +1 to get upper limit, -1 to get lower limit
    :param w: Bremsstrahlung photon energy
    :param mw: WIMP mass
    :param v: WIMP speed (earth/detector frame)
    """
    return (wr.mu_nucleus(mw, material)**2 * v**2 / wr.mn(material=material)
            * (1
               - vmin_w(w, mw, material)**2 / (2 * v**2)
               + sign * (1 - vmin_w(w, mw, material)**2 / v**2)**0.5))


def sigma_w_erec(w, erec, v, mw, sigma_nucleon,
                 interaction='SI', m_med=float('inf')):
    """Differential WIMP-nucleus Bremsstrahlung cross section.
    From Kouvaris/Pradler [arxiv:1607.01789v2], eq. 8

    :param w: Bremsstrahlung photon energy
    :param mw: WIMP mass
    :param erec: recoil energy
    :param v: WIMP speed (earth/detector frame)
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    Default is 'SI' (spin-independent)
    :param m_med: Mediator mass. If not given, assumed very heavy.

    TODO: check for wmax!    # What is this? Still relevant?
    """
    # X-ray form factor
    form_atomic = np.abs(f1(w / nu.keV) + 1j * f2(w / nu.keV))

    # Note mn -> mn c^2, Kouvaris/Pradtler and McCabe use natural units
    return (4 * nu.alphaFS / (3 * np.pi * w) *
            erec / (wr.mn() * nu.c0**2) *
            form_atomic**2 *
            wr.sigma_erec(erec, v, mw, sigma_nucleon, interaction, m_med))


def sigma_w(w, v, mw, sigma_nucleon,
            material,
            interaction='SI', m_med=float('inf'),
            **kwargs):
    """Differential Bremsstrahlung WIMP-nucleus cross section

    :param w: Bremsstrahlung photon energy
    :param v: WIMP speed (earth/detector frame)
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP-nucleon cross-section
    :param interaction: string describing DM-nucleus interaction.
    Default is 'SI' (spin-independent)
    :param m_med: Mediator mass. If not given, assumed much heavier than mw.

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).

    """
    def integrand(erec):
        return sigma_w_erec(w, erec, v, mw, sigma_nucleon, interaction, m_med)

    return quad(integrand,
                erec_bound(-1, w, v, mw, material=material),
                erec_bound(+1, w, v, mw, material=material),
                **kwargs)[0]


@export
@wr.vectorize_first
def rate_bremsstrahlung(w, mw, sigma_nucleon, interaction='SI',
                        m_med=float('inf'), t=None,
                        material='Xe',
                        halo_model=None, **kwargs):
    """Differential rate per unit detector mass and recoil energy of
    Bremsstrahlung elastic WIMP-nucleus scattering.

    :param w: Bremsstrahlung photon energy
    :param mw: Mass of WIMP
    :param sigma_nucleon: WIMP/nucleon cross-section
    :param m_med: Mediator mass. If not given, assumed very heavy.
    :param t: A J2000.0 timestamp. If not given,
    a conservative velocity distribution is used.
    :param halo_model: class (default to standard halo model)
    containing velocity distribution
    :param interaction: string describing DM-nucleus interaction.
    See sigma_erec for options
    :param progress_bar: if True, show a progress bar during evaluation
    (if w is an array)

    Further kwargs are passed to scipy.integrate.quad numeric integrator
    (e.g. error tolerance).
    """
    halo_model = wr.StandardHaloModel() if halo_model is None else halo_model
    vmin = vmin_w(w, mw, material)

    if vmin >= wr.v_max(t, halo_model.v_esc):
        return 0

    def integrand(v):
        return (sigma_w(w, v, mw, sigma_nucleon,
                        interaction=interaction,
                        m_med=m_med,
                        material=material) *
                v * halo_model.velocity_dist(v, t))

    return halo_model.rho_dm / mw * (1 / wr.mn()) * quad(
        integrand,
        vmin,
        wr.v_max(t, halo_model.v_esc),
        **kwargs)[0]


# TODO: change to dblquad instead of 2x single quad!

@export
@wr.vectorize_first
def rate_bremsstrahlungdf(erec, mw, sigma_nucleon, interaction='SI',
                 m_med=float('inf'), t=None, material='Xe',
                 df=None, Potential=None,af=None,posSun=None,v_0=None,v_esc=None,
                   **kwargs):
    if not agama:
        print("agama needs to be installed to run function\n")
        return 0
    t1=t
    if t1 is None:
        t1=59.37
    if posSun is None:
        posSun=np.array([8.27,0,0.025])
    if df is None or Potential is None:
        print("Must supply a df and potential")
        return 0
    Rsun=(posSun[0]**2+posSun[1]**2)**0.5
    if af is None:
        af=agama.ActionFinder(Potential)
    if v_0 is None:
        v_0=np.sqrt(Rsun*(-Potential.force([Rsun,0,0])[0]))
    if v_esc is None:
        v_esc=np.sqrt(-2*Potential.potential(posSun))
    ve=wr.earth_velocity(t1,v_0*nu.km/nu.s)*nu.s/nu.km
    v_min = vmin_w(erec, mw, material)*nu.s/nu.km
    v_max=wr.v_max(t, v_esc*nu.km/nu.s,v_0=v_0*nu.km/nu.s)*nu.s/nu.km
    if(v_max<v_min):
        return 0
    fac=nu.Msolar/(nu.kpc)**3/(mw*wr.mn())
    try:
        len(erec)
    except TypeError:
        def S0(v):
            if(v<v_min or v>v_max):
                return 0
            return fac*sigma_w(erec, v*nu.km/nu.s, mw, sigma_nucleon,
                            interaction, m_med, material=material)*v*nu.km/nu.s
        def sf(xv):
            v=np.array((((xv[:,3]-ve[0])**2+(xv[:,4]-ve[1])**2+(xv[:,5]-ve[2])**2))**0.5)
            try:
                len(v)
            except TypeError:
                return S0(v)
            result=np.zeros(len(v))
            for i in range(len(v)):
                result[i]=S0(v[i])
            return result
        gm=agama.GalaxyModel(potential=Potential,af=af,df=df,sf=sf)
        return gm.moments(posSun,vel2=False)
    else:
        restot=np.zeros(len(erec))
        for j in range(len(erec)):
            def S0(v):
                if(v<v_min or v>v_max):
                    return 0
                return fac*sigma_w(erec[j], v*nu.km/nu.s, mw, sigma_nucleon,
                                interaction, m_med, material=material)*v*nu.km/nu.s
            def sf(xv):
                v=np.array((((xv[:,3]-ve[0])**2+(xv[:,4]-ve[1])**2+(xv[:,5]-ve[2])**2))**0.5)
                try:
                    len(v)
                except TypeError:
                    return S0(v)
                result=np.zeros(len(v))
                for i in range(len(v)):
                    result[i]=S0(v[i])
                return result
            gm=agama.GalaxyModel(potential=Potential,af=af,df=df,sf=sf)
            restot[j]=gm.moments(posSun,vel2=False)
        return restot
    
