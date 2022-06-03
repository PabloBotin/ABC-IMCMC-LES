# There are several distance computing methods I could use 
# A. Summatory of the normalized difference of each wavenumber (only those corresponding to the LES resolved scales).
# B. MSE of log of pdf of S' and S.
# C. Kullbacl-Lieber (KL) divergence of S' and S.
# Ask Peter which one should I use. 
# Meanwhile, prepare the methods for computing it in the 3 proposed ways. 
# B. Mean Square Error (MSE). Production's distance is always computed with this technique.
# B. Applied to Sigma 
# Is the log necessary? I mean, it is part of the MSE or is it assuming that S is the pdf straight? 

# Peter's way:
def distance (S_LES, S_DNS):
    d= np.sqrt(np.absolute(np.log(S_DNS)-np.log(S_LES))**2)


def d_MSE_sigma (S_LES, S_DNS):
    d = 0
    for i in range (3):     # For Sigma11, Sigma12 and Sigma13
        d =+ (S_LES - S_DNS)**2
    return d


# B. Applied to Production 
def d_MSE_P (S_LES, S_DNS):
    d = (S_LES - S_DNS)**2

# C. Kullbalc-Lieber (KL) divergence 
# I do not understand what is Sf, I ll take it as the ref summary statistic btm. 
# Also, I would have to compute the inverse of np.log(). Ask Peter.
def d_KL_sigma (S_LES, S_DNS):
    d = 0
    for i in range (3):     # For Sigma11, Sigma12 and Sigma13
        d =+ np.log.inv(S_LES) * np.absolute(S_LES - S_DNS)
    return d
