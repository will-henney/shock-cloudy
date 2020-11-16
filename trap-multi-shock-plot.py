models=[["1.22", "16.8", "12.6", "4e+03", "5.33e+03", "1e+04", "1.04e+04", "8.23e-05", "6.38"], ["1.41", "19.4", "12.1", "3e+03", "4.8e+03", "1e+04", "1.2e+04", "7.89e-05", "6.35"], ["1.73", "23.8", "11.9", "2e+03", "4e+03", "1e+04", "1.49e+04", "7.57e-05", "6.22"], ["2", "27.5", "12", "1.5e+03", "3.43e+03", "1e+04", "1.77e+04", "7.37e-05", "6"], ["2.45", "33.6", "12.6", "1e+03", "2.67e+03", "1e+04", "2.31e+04", "7.15e-05", "5.54"], ["2.83", "38.8", "13.4", "750", "2.18e+03", "1e+04", "2.85e+04", "6.96e-05", "5.1"], ["3.46", "47.6", "14.9", "500", "1.6e+03", "1e+04", "3.92e+04", "7.01e-05", "4.61"], ["4.47", "61.4", "17.7", "300", "1.04e+03", "1e+04", "6.05e+04", "7.28e-05", "4.03"], ["5.48", "75.2", "20.7", "200", "727", "1e+04", "8.71e+04", "9.65e-05", "4.56"], ["7.75", "106", "27.9", "100", "381", "1e+04", "1.67e+05", "0.000523", "18.3"], ["11", "150", "38.5", "50", "195", "1e+04", "3.26e+05", "0.00561", "142"]]
import os
from pathlib import Path
import numpy as np
from scipy import interpolate, optimize, integrate
from astropy.table import Table
from matplotlib import pyplot as plt
import seaborn as sns
import sys
sys.path.append('/Users/will/Work/CLOUDY/claudia/')
from claudia import CloudyModel

CloudyModel.skipsaves.append('continuum')
CloudyModel.skipsaves.remove(".tim")


k = 1.3806503e-16                         # Boltzmann's constant [cgs]
mp = 1.67262158e-24                       # Proton rest mass [cgs]
yHe = 0.087                              # He/H abundance
xHeplus = 1.0                             # He is all singly ionized
xH = 1.0                                  # H is all ionized
mu = 1.0 + 3.0*yHe                        # Mean mass per nucleon
gamma = 5./3.                             # adiabatic index
yr = 3.15576e7                            # Year in seconds
pc = 3.085677582e18                       # Parsec in cm
km = 1e5

logRadius = 17.5                     # log10 Radius, cm

# Set up graph for temperature and density
pltfile = 'trap-multi-shock-distance.pdf'
fig, (axtop, axbot) = plt.subplots(2, 1, sharex=True)

pltfile_em = pltfile.replace('distance', 'em-distance')
fig_em, axes_em = plt.subplots(9, 1, sharex=True)
fig_em.set_size_inches(10, 27)
ax6563, ax5007, ax4363, axO3Ha, axLcool, ax5007frac, axOcharge, axTagain, axNagain = axes_em

pltfile_em2 = pltfile.replace('distance', 'em2-distance')
fig_em2, (ax6563_2, ax5007_2) = plt.subplots(2, 1, sharex=True)

pltfile_emcum = pltfile.replace('distance', 'emcum')
fig_cum, [
    ax5007_cum,
    ax4363_cum,
    axLc_cum,
    axO3frac_cum,
    axOcharge_cum,
    axs_cum,
    axden_cum,
    axT_cum,
] = plt.subplots(8, 1, sharex=True, figsize=(6, 16))

# Loop over all the shock velocities
colors = sns.dark_palette('orange', len(models[:-1]))
for row, c in reversed(list(zip(models[:-1], colors))):
    M0, u0, v1, n0, n1, N2, T1, dcool, tcool = [float(x) for x in row]
    model_id = f'trap-shock-R{int(10*logRadius):d}-shock-v{u0:03.0f}'
    label = 'Vs = {:.0f} km/s'.format(u0)

    try:
        m = CloudyModel(str(Path("models") / model_id), niter=0)
    except:
        print('Failed to read', model_id)
        continue
    # Net cooling coefficient for all times
    NeNp = m.ovr.HII*m.ovr.hden*m.ovr.eden
    Lambda_full = (m.cool.Ctot_ergcm3s - m.cool.Htot_ergcm3s)/NeNp
    # index corresponding to initial post-shock state
    # Heuristic is that it is point where net cooling is highest
    istart = np.argmax(Lambda_full)
    # And corresponding T, which should be more or less T1
    Tstart = m.cool.Temp_K[istart]
    # Photoionization equilibrium T
    Teq = m.cool.Temp_K.min()
    print(istart, Teq, Tstart)
    # Now restrict to the post-shock zone
    T_grid = m.cool.Temp_K[istart:]
    Lambda_grid = Lambda_full[istart:]
    integrand_grid = T_grid**2 / Lambda_grid
    integral_grid = integrate.cumtrapz(integrand_grid, T_grid, initial=0.0)
    T = T_grid
    s = (2./3.)*(Lambda_grid[0]/Tstart**3)*(integral_grid[0] - integral_grid)

    # We need to recalculate tcool and dcool because the Lambda(T1) is
    # now very different - it is much higher because of the under-ionization
    Lambda1 = Lambda_grid[0]
    Pressure = (m.ovr.hden*(1.0 + yHe) + m.ovr.eden)*k*m.cool.Temp_K
    Pressure *= n1/N2
    NeNp *= (n1/N2)**2
    P1 = Pressure[istart]
    L1 = Lambda1*NeNp[istart]
    # Cooling time in seconds
    tcool = P1/((gamma - 1.)*L1)
    # Cooling distance in parsecs
    dcool = v1*km*tcool/pc

    x = np.hstack([[-0.05, 0.0], dcool*s]) 
    axtop.semilogy(x, np.hstack([[Teq, Teq], T]), color=c)
    den = n1*Tstart/T
    axbot.semilogy(x, np.hstack([[n0, n0], den]), label=label, color=c)

    # And plot the emissivities too
    Lcool = m.cool.Ctot_ergcm3s[istart:]*(den/N2)**2
    em5007 = (m.ems.O__3_500684A[istart:])*(den/N2)**2 
    em4363 = (m.ems.O__3_436321A[istart:])*(den/N2)**2 
    em6563 = (m.ems.H__1_656285A[istart:])*(den/N2)**2 
    Ostack = np.vstack([m.ovr["O"+j] for j in "123456"])
    O789 = 1.0 - Ostack.sum(axis=0)
    Ostack = np.vstack([m.ovr["O"+j] for j in "123456"] + [O789])
    Ocharge = np.sum(Ostack*np.arange(7)[:, None], axis=0)[istart:]
    istop = np.nanargmax(s[T > 1.01*Teq])
    ss = s/s[istop]

    # Fractional cumulative emissivity of [O III]
    cumem = integrate.cumtrapz(em5007, s*dcool, initial=0.0)
    tot5007 = cumem[istop]
    cumem = tot5007 - cumem
    #cumem /= cumem[istop]

    T0 = np.average(T[:istop], weights=em5007[:istop])
    t2 = np.average(((T[:istop]-T0)/T0)**2, weights=em5007[:istop])
    tlabel = f"{label} $T = {T0/1e3:.1f}$ kK, $t^2 = {t2:.3f}$"

    ax5007.plot(ss, em5007, color=c)
    ax6563.plot(ss, em6563, color=c)
    ax5007_2.plot(ss, em5007, label=label, color=c)
    ax6563_2.plot(ss, em6563, color=c)
    ax4363.plot(ss, em4363/em5007, label=label, color=c)
    axO3Ha.plot(ss, em5007/em6563, color=c)
    axLcool.plot(ss, Lcool, color=c)
    ax5007frac.plot(ss, em5007/Lcool, color=c)
    axOcharge.plot(ss, Ocharge, color=c)
    axTagain.plot(ss, T, color=c)
    axNagain.plot(ss, den, color=c)

    xx = dcool*(s[istop]-s)
    ax5007_cum.plot(xx, em5007, label=tlabel, color=c)
    ax4363_cum.plot(xx, em4363/em5007, color=c)
    axLc_cum.plot(xx, Lcool, color=c)
    axO3frac_cum.plot(xx, em5007/Lcool, color=c)
    axOcharge_cum.plot(xx, Ocharge, color=c)
    axs_cum.plot(xx, cumem, color=c)
    axden_cum.plot(xx, den, color=c)
    axT_cum.plot(xx, T, color=c)


axtop.set_ylim(5000, 0.5e6)
axbot.set_ylim(30.0, 2e4)
axbot.set_xlim(-1.1e-5, 9e-5)
axbot.set_xlabel('Distance, pc')
axbot.set_ylabel('Density, pcc')
axtop.set_ylabel('Temperature, K')
axbot.legend(ncol=2, fontsize='x-small', loc='upper left')
fig.savefig(pltfile)

axes_em[-1].set_xlabel('Fraction of total cooling distance')
ax6563.set_ylabel('Hα 6563 emissivity')
ax4363.legend(ncol=2, fontsize='x-small', loc='lower left')
ax4363.set_ylabel('[O III] 4363/5007 ratio')
axO3Ha.set_ylabel('[O III] 5007/Hα ratio')
ax5007.set_ylabel('[O III] 5007 emissivity')
axLcool.set_ylabel('Total cooling, erg/cm³/s')
axTagain.set_ylabel('Temperature, K')
axNagain.set_ylabel('Total Hydrogen density, /cm³')
ax5007frac.set_ylabel('[O III] 5007 fraction of cooling')
axOcharge.set_ylabel('Mean charge of Oxygen')
for ax in axes_em:
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlim(0.0, 1.2)
#ax5007.set_ylim(3e-25, 1.5e-20)
#axO3Ha.set_ylim(0.1, 150)
axOcharge.set_yscale('linear')
axOcharge.set_ylim(0.0, 8.0)
for ax in axLcool, ax4363, ax5007, ax6563, axNagain, axTagain, ax5007frac, axO3Ha:
    ax.set_yscale('linear')
    ax.set_ylim(0.0, None)
axO3Ha.set_ylim(0.0, 40.0)



fig_em.tight_layout()
fig_em.savefig(pltfile_em)


ax5007_2.set_ylim(0.0, None)
ax6563_2.set_ylim(0.0, None)
ax5007_2.set_xlabel('Fraction of total cooling distance')
ax5007_2.set_ylabel('[O III] 5007 emissivity')
ax6563_2.set_ylabel('Hα 6563 emissivity')
ax5007_2.set_xlim(0.0, 1.2)
ax5007_2.legend(ncol=2, fontsize='x-small', loc='upper left')
fig_em2.savefig(pltfile_em2)

ax5007_cum.set_xlim(-1e-5, 6.5e-5)
ax5007_cum.set_ylim(0.0, None)
ax4363_cum.set_ylim(0.0, None)
axLc_cum.set_ylim(0.0, None)
axOcharge_cum.set_ylim(1.9, 3.1)
axs_cum.set_ylim(0.0, None)
axT_cum.set_ylim(0.0, 15e4)
ax5007_cum.legend(ncol=2, fontsize='x-small', loc='lower right')
axT_cum.set_xlabel("Distance from equilibrium shell")
ax5007_cum.set_ylabel('[O III] 5007')
ax4363_cum.set_ylabel('[O III] 4363 / 5007')
axden_cum.set_ylabel('Density, pcc')
axT_cum.set_ylabel('Temperature, K')
axLc_cum.set_ylabel("Cooling, erg/cm³/s")
axO3frac_cum.set_ylabel("5007 cool frac")
axOcharge_cum.set_ylabel('Mean O charge')
axs_cum.set_ylabel('Cumulative 5007')
fig_cum.tight_layout()
fig_cum.savefig(pltfile_emcum)
