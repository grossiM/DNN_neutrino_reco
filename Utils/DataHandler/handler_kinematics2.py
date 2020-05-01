import numpy as np
import sys
import math

from ROOT import TLorentzVector

def abcdelta_iteration(l_px, l_py, l_pz, l_e, v_px, v_py, mW=80.385):

    lpxvp = l_px*v_px + l_py*v_py
    met = math.sqrt(v_px**2 + v_py**2)
    met2 = met**2

    a = l_pz**2 - l_e**2
    b = mW**2*l_pz + 2*l_pz*lpxvp
    c = mW**4/4 + (lpxvp + mW**2)*lpxvp - l_e**2*met2
    delta = b**2 - 4*a*c
    if delta < 0:
        delta = 0

    return a, b, c, delta

def abcdelta( pdarray, flavour):
    lv_variables = [flavour+'_py', flavour+'_pz', flavour+'_E', 'v_'+flavour+'_px', 'v_'+flavour +'_py']
    try:
        lv = np.expand_dims(pdarray[flavour+'_px'],1)
        for variable in lv_variables: lv = np.append(lv,np.expand_dims(pdarray[variable],1),1)
    except:
        print('Error: ',flavour,' + v_',flavour,' neutrino pair not found in input dataset.',sep="")
        sys.exit(1)

    vectorized_abcdelta = np.vectorize(abcdelta_iteration)
    a, b, c, delta = vectorized_abcdelta(lv[:,0],lv[:,1],lv[:,2],lv[:,3],lv[:,4],lv[:,5])
    return a, b, c, delta


def tag_solutions_iteration( a, b, delta, v_pz):
    
    sol0 = (-b - math.sqrt(delta))/2/a
    sol1 = (-b + math.sqrt(delta))/2/a

    if math.fabs(sol0 - v_pz) < math.fabs(sol1 - v_pz):
        label = 0
    else:
        label = 1

    return sol0, sol1, label

def tag_solutions( pdarray, flavour):
    try:
        a = pdarray[flavour+'_a']
        b = pdarray[flavour+'_b']
        delta = pdarray[flavour+'_delta']
    except:
        print('Error: a, b and delta do not exist. Run appendQuadEqParams first.')
        sys.exit(1)

    try:
        v_pz = pdarray['v_'+flavour+'_pz']
    except:
        print(flavour+'v_pz branch does not exist.')
        sys.exit(1)

    tag_solutions_vectorized = np.vectorize(tag_solutions_iteration)
    sol0, sol1, label = tag_solutions_vectorized( a, b, delta, v_pz)
    return sol0, sol1, label

def cos_theta_iterate(lp, vp):
    Wp = lp + vp
    lp.Boost(-Wp.BoostVector())
    return lp.Vect().Unit().Dot(Wp.Vect().Unit())


def cos_theta_all(meas):

    n_events = meas.shape[0]
    cosTheta = np.zeros((n_events,))

    for event in range(n_events):

        lpx = meas[event][0]
        lpy = meas[event][1]
        lpz = meas[event][2]
        le = meas[event][3]

        lp = TLorentzVector()
        lp.SetPxPyPzE(lpx, lpy, lpz, le)

        vpx = meas[event][4]
        vpy = meas[event][5]
        vpz = meas[event][6]
        vpe = meas[event][7]

        vp = TLorentzVector()
        vp.SetPxPyPzE(vpx, vpy, vpz, vpe)

        cosTheta[event] = cos_theta_iterate(lp, vp)

    return cosTheta

def cos_theta(pdarray, variables):
    meas = pdarray[variables]
    return cos_theta_all(meas.values)

def calc_energy_iterate(px, py, pz):
    return math.sqrt(px**2 + py**2 + pz**2)

def calc_energy(threeVector):
    vcalc_energy_iterate = np.vectorize(calc_energy_iterate)
    return vcalc_energy_iterate(threeVector[:,0],threeVector[:,1],threeVector[:,2])

def calc_mass_iterate(px, py, pz, e):
    return math.sqrt(abs(e**2 - px**2 - py**2 - pz**2))

calc_mass = np.vectorize(calc_mass_iterate)

#########################################################################################new part
def vector_manipulation_iterate(lp, vp):
    W = lp + vp
    
    #return vplus.Dot(Wplus)
    return np.dot(vp, W)


def vector_manipulation_all(meas):

    n_events = meas.shape[0]
    pL = np.zeros((n_events,))

    for event in range(n_events):

        lpx = meas[event][0]
        lpy = meas[event][1]
        lpz = meas[event][2]

        lp = np.array([lpx, lpy, lpz])

        vpx = meas[event][3]
        vpy = meas[event][4]
        vsol = meas[event][5]

        vp = np.array([vpx, vpy, vsol])

        pL[event] = vector_manipulation_iterate(lp, vp)
        
    return pL

def vector_manipulation(pdarray, variables):
    meas = pdarray[variables]
    return vector_manipulation_all(meas.values)

def tag_selectioncriteria_iteration( a, b, pLplus, pLminus, v_pz, v_mu_sol0, v_mu_sol1):
    
    #sign =  round(np.random.uniform(0,1))*2 -1
    #random = (-b + sign*math.sqrt(delta))/2/a
    random = round(np.random.uniform(0,1))
    ######## sel 1 ###########
    if ((pLplus < 5000 and pLminus < 5000) or (pLplus > 5000 and pLminus > 5000) ):
        sel1 = random
    elif (pLplus < 5000 and pLminus > 5000):
        sel1 = 0
    elif (pLplus > 5000 and pLminus < 5000):
        sel1 = 1
    else: print(' sel 1 wrong case')
    ######## sel 2 ###########
    if((math.fabs(v_mu_sol0) < 50 and math.fabs(v_mu_sol1) < 50) or (math.fabs(v_mu_sol0) > 50 and math.fabs(v_mu_sol1) > 50)):
        sel2 = random
    elif(math.fabs(v_mu_sol0) < 50 and math.fabs(v_mu_sol1) > 50):
        sel2 = 1
    elif(math.fabs(v_mu_sol0) > 50 and math.fabs(v_mu_sol1) < 50):
        sel2 = 0
    else: print('sel 2 wrong case')
    ######## sel 3 ###########
    if((math.fabs(pLplus*a/b) < 25 and math.fabs(pLminus*a/b) < 25) or (math.fabs(pLplus*a/b) > 25 and math.fabs(pLminus*a/b) > 25)):
        sel3 = random
    elif (math.fabs(pLplus*a/b) > 25 and math.fabs(pLminus*a/b) < 25):
        sel3 = 1
    elif (math.fabs(pLplus*a/b) < 25 and math.fabs(pLminus*a/b) > 25):
        sel3 = 0
    else:
        print('sel 3 wrong case')
    ######## sel 4 ###########
    if((-v_mu_sol0*a/b <= 0.5 and -v_mu_sol1*a/b <= 0.5) or (-v_mu_sol0*a/b > 0.5 and -v_mu_sol1*a/b > 0.5)):
        sel4 = random
    elif(-v_mu_sol0*a/b < 0.5 and -v_mu_sol1*a/b > 0.5):
        sel4 = 1
    elif(-v_mu_sol0*a/b > 0.5 and -v_mu_sol1*a/b < 0.5):
        sel4 = 0
    else:
        print('DD sel 4 wrong case')
    ######## sel 5 ###########
    if(math.fabs(v_mu_sol0) < 50 and math.fabs(v_mu_sol1) < 50):
        if((math.fabs(pLplus*a/b) < 25 and math.fabs(pLminus*a/b) < 25) or (math.fabs(pLplus*a/b) > 25 and math.fabs(pLminus*a/b) > 25)):
            sel5 = random
        elif (math.fabs(pLplus*a/b) > 25 and math.fabs(pLminus*a/b) < 25):
            sel5 = 1
        elif (math.fabs(pLplus*a/b) < 25 and math.fabs(pLminus*a/b) > 25):
            sel5 = 0
        else:
            print('sel 5a wrong case')
    elif(math.fabs(v_mu_sol0) > 50 and math.fabs(v_mu_sol1) > 50):
        if(math.fabs(pLplus*a/b)>math.fabs(pLminus*a/b)):
            sel5 = 1
        else:
            sel5 = 0
    elif(math.fabs(v_mu_sol0) < 50 and math.fabs(v_mu_sol1) > 50):
        sel5 = 1
    elif(math.fabs(v_mu_sol0) > 50 and math.fabs(v_mu_sol1) < 50):
        sel5 = 0
    else:
        print('sel 5b wrong case')

    return sel1,sel2,sel3,sel4,sel5
#1
def tag_selectioncriteria( pdarray, flavour):
    try:
        a = pdarray[flavour+'_a']
        b = pdarray[flavour+'_b']
        pLplus = pdarray['pL'+'plus']
        pLminus = pdarray['pL'+'minus']
        v_mu_sol0 = pdarray['v_mu_sol0']
        v_mu_sol1= pdarray['v_mu_sol1']
        
    except:
        print('Error: a,b,pLplus,pLminus, v_mu_sol0, v_mu_sol1 do not exist. Run calcpL first.')
        sys.exit(1)
    try:
        v_pz = pdarray['v_'+flavour+'_pz']
    except:
        print(flavour+'v_pz branch does not exist.')
        sys.exit(1)

    tag_selectioncriteria_vectorized = np.vectorize(tag_selectioncriteria_iteration)
    sel1,sel2,sel3,sel4,sel5 = tag_selectioncriteria_vectorized( a, b, pLplus, pLminus, v_pz, v_mu_sol0, v_mu_sol1)

    return sel1,sel2,sel3,sel4,sel5

