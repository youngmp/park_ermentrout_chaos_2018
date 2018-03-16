"""
generate a table of distribution values for use in xpp (thetan_het.ode)

Thi script runs a random number generator to create N random numbers drawn from some distribution.

it then appends 3 terms to the beginning of the data array:
npts (number of data points)
xlo (lowest x-value, in this case 0)
xhi (highest x-value, in this case N-1)

the resulting array is saved to a tab file and can be directly imported into xpp.

"""
import getopt
import sys

from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(suppress=True)
np.set_printoptions(precision=7)

np.random.seed(0)


def usage():
    print "-c, --cauchy\t\t: use cauchy distribution"
    print "-n, --normal\t\t: use normal distribution"
    print "-u, --uniform\t\t: use uniform distribution"


def generate_data(choice,N=100,a=-2,b=2):
    """
    choice: distribution type
    N: total number of random numbers to draw
    a,b: lower and upper bounds for truncated distribution
    """
    if choice == 'cauchy':
        s = np.random.standard_cauchy(N)

    elif choice == 'cauchy-trunc':
        s = np.nan
    
    elif choice == 'normal':
        s = np.random.randn(N)

    elif choice == 'normal-trunc':
        s = truncnorm.rvs(a,b,size=N)

        
    elif choice == 'uniform':
        s = np.random.rand(N)

    #s = s[(s>-bounds)*(s<bounds)]
    return s

def generate_table(data):
    """
    append 3 terms to the beginning of the data file:
    npts
    xlo
    xhi

    choose xlo=0, xhi=len(data)-1
    """
    new_data = np.zeros(len(data)+3)
    
    new_data[0] = len(data)
    new_data[1] = 0
    new_data[2] = len(data)-1

    new_data[3:] = data
    
    return new_data

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "cCnNu", ["cauchy","cauchy-trunc","normal","normal-trunc","uniform"])

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()

        else:
            if opt in ("-c","--cauchy"):
                choice = 'cauchy'
            if opt in ("-C","--cauchy-trunc"):
                choice = 'cauchy-trunc'
            elif opt in ('-n','--normal'):
                choice = 'normal'
            elif opt in ('-N','--normal-trunc'):
                choice = 'normal-trunc'
            elif opt in ('-u','--uniform'):
                choice = 'uniform'

    N = 300

    data1 = generate_data(choice,N=N)
    data2 = generate_data(choice,N=N)

    table1 = generate_table(data1)
    table2 = generate_table(data2)

    np.savetxt('eta1_N='+str(N)+'.tab',table1,fmt='%.14f')
    np.savetxt('eta2_N='+str(N)+'.tab',table2,fmt='%.14f')

if __name__ == "__main__":
    main(sys.argv[1:])
