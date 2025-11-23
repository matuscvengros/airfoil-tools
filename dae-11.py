import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def cst_airfoil(upper_weights, lower_weights, le_weight, te_thickness, num_points=400):
    # 1. Cosine Spacing
    beta = np.linspace(0, np.pi, num_points)
    psi = 0.5 * (1.0 - np.cos(beta))
    
    # 2. Class Function
    C = np.sqrt(psi) * (1 - psi)
    
    # 3. Shape Function S(psi)
    n_order = len(upper_weights) - 1
    
    def get_S(w, p_vals):
        S = np.zeros_like(p_vals)
        for i in range(n_order + 1):
            K = comb(n_order, i)
            basis = K * (p_vals**i) * ((1 - p_vals)**(n_order - i))
            S += w[i] * basis
        return S

    S_upper = get_S(upper_weights, psi)
    S_lower = get_S(lower_weights, psi)
    
    # 4. AeroSandbox LE Correction (Same for both surfaces)
    N_weights = len(upper_weights)
    le_term = le_weight * psi * (1 - psi)**(N_weights + 0.5)

    # 5. Final Coordinate Calculation
    z_upper = C * S_upper + le_term + psi * (te_thickness / 2.0)
    z_lower = C * S_lower + le_term - psi * (te_thickness / 2.0)
    
    return psi, z_upper, z_lower

# --- EXACT WEIGHTS ---
LE_Weight = 0.5035068867316306
TE_Thick = 0.00011059192135088117

Upper = [0.17036014, 0.15272658, 0.51688551, 0.09212467, 0.66904449, 0.14345864, 0.28990386, 0.16207357]
Lower = [-0.16310508, -0.14398528, 0.08896135, -0.0706084, 0.09741634, 0.01466031, 0.07888292, 0.08075122]

# --- GENERATE ---
psi, z_upper, z_lower = cst_airfoil(Upper, Lower, LE_Weight, TE_Thick)

# --- PLOT (EQUAL ASPECT RATIO) ---
plt.figure(figsize=(12, 4)) # Wide figure
plt.plot(psi, z_upper, color='purple', linewidth=2, label='Upper')
plt.plot(psi, z_lower, color='red', linewidth=2, label='Lower')
plt.fill_between(psi, z_upper, z_lower, color='lavender', alpha=0.5)

plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title('DAE-11 (Equal Scale)')
plt.xlabel('x/c')
plt.ylabel('z/c')

# Force Equal Aspect Ratio (1 unit x = 1 unit y)
plt.gca().set_aspect('equal', adjustable='box')

plt.grid(True, alpha=0.3)
plt.legend()
plt.show()