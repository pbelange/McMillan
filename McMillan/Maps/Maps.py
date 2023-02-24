import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
sp.init_printing(use_unicode=True,fontsize=24)

from collections import OrderedDict


class Map1D():
    def __init__(self, q_map,p_map,q,p):


        # Base Variables
        #-----------------------
        self.q,self.p          = q,p
        self.q_map,self.p_map  = q_map,p_map
        self.jacobian          = sp.Matrix([[self.q_map.diff(self.q),self.q_map.diff(self.p)],
                                            [self.p_map.diff(self.q),self.p_map.diff(self.p)]])


        # Flouquet-coordinates
        
        #-----------------------

        # Numerical Iteration
        #-----------------------
        self.q_iter = sp.lambdify((self.q,self.p),self.q_map)
        self.p_iter = sp.lambdify((self.q,self.p),self.p_map)
        #-----------------------


        # Constants for perturbation expansion
        #-----------------------
        self.C     = sp.IndexedBase('C')
        self.eps   = sp.symbols('epsilon',real=True)
        self.C_cs  = sp.symbols('C_cs',real=True)
        #-----------------------

        # Perturbation Expansion, (q,p) -> (eps*q,eps*p)
        #-----------------------
        self.q_eps             = (1/self.eps*q_map.subs({self.q:self.eps*self.q,
                                                        self.p:self.eps*self.p},simultaneous=True)).expand()
                                                
        self.p_eps             = (1/self.eps*p_map.subs({self.q:self.eps*self.q,
                                                        self.p:self.eps*self.p},simultaneous=True)).expand()
        #-----------------------

        
        # Creating the K's
        #-----------------------
        self.max_order = 10
        for n in range(self.max_order+1):
            self.__setattr__(f'K{n}',self.make_K(self.lex_power(n)))
        #-----------------------


    def lex_power(self,order):
        # ex for K1 : [(3,0),(2,1),(1,2),(0,3)] -> x**3 + x**2*y + x*y**2 + y**3
        return [(m,n) for n in range(13) for m in range(13) if n+m==order+2]


    def cycle(self,q0,p0,n_cycles):
        # To symbolically cycle the map
        q_prime,p_prime = q0,p0
        for i in range(n_cycles):
            q_prime,p_prime = self.q_map.subs({self.q:q_prime,self.p:p_prime},simultaneous=True),self.p_map.subs({self.q:q_prime,self.p:p_prime},simultaneous=True)
        return q_prime,p_prime

    def iterate(self,q0,p0,n_turns):
        # To numerically iterate the map
        q_vec,p_vec = np.nan*np.ones(n_turns),np.nan*np.ones(n_turns)
        q_vec[0],p_vec[0] = q0,p0
        for ii in range(n_turns-1):
            q_vec[ii+1],p_vec[ii+1] = self.q_iter(q_vec[ii],p_vec[ii]),self.p_iter(q_vec[ii],p_vec[ii])
        return q_vec,p_vec
        
        
    def truncate(self,expr,order):
        return sum(coeff*self.q**pwr[0]*self.p**pwr[1] for pwr,coeff in sp.Poly(expr,self.q,self.p).terms() if pwr[0]+pwr[1]<=order)

    def make_K(self,powers):
        expr   = sum(self.C[pwr[0],pwr[1]]*self.q**pwr[0]*self.p**pwr[1] for pwr in powers)
        return expr

    def Kofn(self,order):
        return sum((self.eps**n)*self.__getattribute__(f'K{n}') for n in range(order+1))

    def residue(self,order):
        K =  self.Kofn(order)
        return (K.subs({self.q:self.truncate(self.q_eps,order+1),
                        self.p:self.truncate(self.p_eps,order+1)},simultaneous=True) - K).expand()


    def solve_C(self,order,verbose=False,auto_update=True,ignore_symplecticity=False):


        to_cancel = self.residue(order).as_coefficients_dict(self.eps)[self.eps**order]
        eq_list = list(to_cancel.as_coefficients_dict(self.p,self.q).values())


        if order == 0:
            # Check that the map is symplectic (or close enough)
            if not ignore_symplecticity:
                assert np.isclose(np.float64(self.jacobian.det().simplify()-1),0, rtol=1e-9,atol=1e-9), 'Map is not symplectic'

            # Analytic solution for K0 to enforce symplecticity
            _coeff_q = self.q_eps.expand().as_coefficients_dict(self.q,self.p)
            _coeff_p = self.p_eps.expand().as_coefficients_dict(self.q,self.p)
            a  = _coeff_q[self.q]
            b  = _coeff_q[self.p]
            c  = _coeff_p[self.q]
            d  = _coeff_p[self.p]

            to_sub = {self.C[1,1]:(b*c-d**2+1)/(b*d) * self.C_cs, self.C[2,0]:-c*self.C_cs/b,self.C[0,2]:self.C_cs}

        else:
            # Solve the system of equations
            to_sub = sp.solve(eq_list,[self.C[i] for i in self.lex_power(order)])

        if verbose:
            print(60*'-')
            for key,item in to_cancel.as_coefficients_dict(self.p,self.q).items():
                display((sp.Matrix([key]),item))
        
        if auto_update:
            self.update(to_sub,verbose=verbose)
        elif verbose:
            display(to_sub)
        
        if verbose:
            print(60*'-')


    def update(self,to_update,verbose=False,deep_update=False):
        if verbose:
            display(to_update)

        for n in range(self.max_order+1):
            # updating the expression
            self.__setattr__(f'K{n}', self.__getattribute__(f'K{n}').subs(to_update, simultaneous=True))

        # Rewrite all variables if needed
        if deep_update:
            for attr in ['q_map','p_map','q_eps','p_eps']:
                self.__setattr__(attr, self.__getattribute__(attr).subs(to_update, simultaneous=True))

            self.q_iter = sp.lambdify((self.q,self.p),self.q_map)
            self.p_iter = sp.lambdify((self.q,self.p),self.p_map)



def plot_invariant(expr,q,p,eval_at = None,**kwargs):
    K = sp.lambdify((q,p),expr)

    

    if eval_at is not None:
        q_list,p_list = eval_at
        q_list = np.array(q_list)
        p_list = np.array(p_list)

        levels = K(q_list,p_list)
        
        kwargs.update({'levels':np.sort(levels)})

        multiplier = 1
        window = ((multiplier *q_list,-multiplier *q_list,multiplier *p_list,-multiplier *p_list))
        Q,P = np.meshgrid(  np.linspace(np.min(window),np.max(window),500),
                            np.linspace(np.min(window),np.max(window),500))

    else:
        Q,P = np.meshgrid(  np.linspace(-10,10,10000),
                            np.linspace(-10,10,10000))
    plt.contour(Q,P,K(Q,P),**kwargs)


