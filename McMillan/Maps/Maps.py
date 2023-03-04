import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import warnings



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

        # Polynomial Expansion
        self.a     = sp.IndexedBase('a',real=True,positive=True)
        self.b     = sp.IndexedBase('b',real=True,positive=True)
        
        try:
            self.a_values = self.q_map.as_poly(self.q,self.p).as_dict()
            self.b_values = self.p_map.as_poly(self.q,self.p).as_dict()

            self.q_poly   = sum(self.a[i,j]*self.q**i*self.p**j for (i,j) in self.a_values.keys())
            self.p_poly   = sum(self.b[i,j]*self.q**i*self.p**j for (i,j) in self.b_values.keys())

            self._is_poly = True
        except:
            self._is_poly = False
            warnings.warn('Map is not a polynomial')
        #-----------------------


        # Floquet-coordinates
        #-----------------------
        if self._is_poly:
            self.alpha,self.beta,self.gamma,self.mu = self.get_twiss()
            self.J,self.phi = sp.symbols('J,phi',real=True,positive=True)

            self.q_flq = sp.sqrt(2*self.beta*self.J)*sp.cos(self.phi)
            self.p_flq =-sp.sqrt(2*self.J/self.beta)*(sp.sin(self.phi) + self.alpha*sp.cos(self.phi))
        else:
            self.alpha,self.beta,self.gamma,self.mu,self.q_flq, self.p_flq = None,None,None,None,None,None
        #-----------------------

        # Numerical Iteration
        #-----------------------
        self.q_iter = sp.lambdify((self.q,self.p),self.q_map)
        self.p_iter = sp.lambdify((self.q,self.p),self.p_map)
        #-----------------------


        # Constants for perturbation expansion
        #-----------------------
        self.C     = sp.IndexedBase('C')
        self.eps   = sp.symbols('epsilon',real=True,positive=True)
        self.C_cs  = sp.symbols('C_cs',real=True,positive=True)
        #-----------------------

        # Perturbation Expansion, (q,p) -> (eps*q,eps*p)
        #-----------------------
        self.q_eps             = (1/self.eps*self.q_map.subs({self.q:self.eps*self.q,
                                                        self.p:self.eps*self.p},simultaneous=True)).expand()
                                                
        self.p_eps             = (1/self.eps*self.p_map.subs({self.q:self.eps*self.q,
                                                        self.p:self.eps*self.p},simultaneous=True)).expand()
        #-----------------------

        
        # Creating the K's
        #-----------------------
        self.num_tol   = 1e-10
        self.max_order = 12
        for n in range(self.max_order+1):
            self.__setattr__(f'K{n}',self.make_K(self.lex_power(n)))
        #-----------------------


    def a_subs(self,ij):
        # Returns the dictionary of a's to be substituted
        return {self.a[_ij]:(self.a_values[_ij] if _ij in self.a_values.keys() else 0) for _ij in ij }

    def b_subs(self,ij):
        # Returns the dictionary of b's to be substituted
        return {self.b[_ij]:(self.b_values[_ij] if _ij in self.b_values.keys() else 0) for _ij in ij }

    def get_twiss(self):
        # Returns the Twiss parameters
        _a,_b,_c,_d = self.a[1,0],self.a[0,1],self.b[1,0],self.b[0,1]
        _subs ={**self.a_subs([(1,0),(0,1)]),**self.b_subs([(1,0),(0,1)])}
        
        _trace = (_b*_c+1)/_d + _d

        _alpha = (_b*_c-_d**2+1)/sp.sqrt(4*(_d**2)-(_d**2)*(_trace**2))
        _beta  = (2*_b*_d)/sp.sqrt(4*(_d**2)-(_d**2)*(_trace**2))
        _mu    = sp.acos(_trace/2)

        if _beta.subs(_subs,simultaneous=True).is_number:
            if _beta.subs(_subs,simultaneous=True) < 0:
                _alpha = -_alpha
                _beta  = -_beta
                _mu    = -_mu + 2*sp.pi
        
          
        return (_alpha.subs(_subs,simultaneous=True),
                _beta.subs(_subs,simultaneous=True),
                ((1+_alpha**2)/_beta).subs(_subs,simultaneous=True),
                _mu.subs(_subs,simultaneous=True))


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


    def cropnum(self,expr):
        return expr.replace(lambda x: x.is_Number and abs(x) < self.num_tol, lambda x: 0)


    def make_K(self,powers):
        expr   = sum(self.C[pwr[0],pwr[1]]*self.q**pwr[0]*self.p**pwr[1] for pwr in powers)
        return expr

    def Kofn(self,order):
        return sum((self.eps**n)*self.__getattribute__(f'K{n}') for n in range(order+1))


    def residue(self,order):
        K =  self.Kofn(order)
        return (K.subs({self.q:self.q_eps,
                        self.p:self.p_eps},simultaneous=True) - K).expand()


    def solve_C(self,order,verbose=False,auto_update=True,force_order_0 = True,ignore_symplecticity=False,at_machine_precision=True):


        to_cancel = self.residue(order).as_coefficients_dict(self.eps)[self.eps**order]
        eq_list = list(to_cancel.as_coefficients_dict(self.p,self.q).values())


        if (force_order_0)&(order == 0):
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
            if (np.mod(order,2)==0)&(order!=0):
                solutions, = sp.linsolve(eq_list[1:],[self.C[i] for i in self.lex_power(order)])
            else:
                solutions, = sp.linsolve(eq_list,[self.C[i] for i in self.lex_power(order)])

            if at_machine_precision:
                # to_sub = {self.C[i]:self.cropnum(s).nsimplify() for i,s in zip(self.lex_power(order),solutions)}
                to_sub = {self.C[i]:self.cropnum(s) for i,s in zip(self.lex_power(order),solutions)}
            else:
                to_sub = {self.C[i]:s for i,s in zip(self.lex_power(order),solutions)}

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


    def solve_C_legacy(self,order,verbose=False,auto_update=True,ignore_symplecticity=False):


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

    
    def minimize_residue(self,order,verbose=False,auto_update=True):

        eps_terms = self.residue(order=order).expand().as_coefficients_dict(self.eps)
        _keys = list(eps_terms.keys())
        if 1 in _keys:
            _keys.remove(1)
        if self.eps in _keys:
            _keys.remove(self.eps)
        lead_pwr = min([key.args[1] for key in _keys if key.args[1]>order])

        _residue = eps_terms[self.eps**lead_pwr]
        _residue = _residue.subs({self.q:self.q_flq,self.p:self.p_flq},simultaneous=True).subs({self.J:1})
        
        # _residue = (self.residue(order=order).expand()+ sp.O(self.eps**(lead_pwr+1))).removeO()
        # _residue = _residue.subs({self.q:self.q_flq,self.p:self.p_flq},simultaneous=True).subs({self.J:1,self.eps:1})
        
        # Cij found in the expression
        _Cij = list((_residue.free_symbols - _residue.atoms(sp.Symbol)))[0]
        _fun = sp.lambdify((self.phi,_Cij),_residue**2)

        def avg_residue(Cij):  
            return scipy.integrate.quad(lambda phi:_fun(phi,Cij),0,2*np.pi)[0]

        # Rounding to 6 decimals to avoid numerical issues
        _value = np.round(scipy.optimize.minimize(avg_residue, x0=0).x[0],6)

        _,i,j = _Cij.args
        to_sub = {self.C[i,j]:_value}
        
        if auto_update:
            self.update(to_sub,verbose=verbose)
        elif verbose:
            display(to_sub)
        if verbose:
            print(60*'-')

        return _value
    

    def minimize_multiple_residue(self,order,verbose=False,auto_update=True):

        eps_terms = self.residue(order=order).expand().as_coefficients_dict(self.eps)
        _keys = list(eps_terms.keys())
        if 1 in _keys:
            _keys.remove(1)
        if self.eps in _keys:
            _keys.remove(self.eps)
        lead_pwr = min([key.args[1] for key in _keys if key.args[1]>order])

        _residue = eps_terms[self.eps**lead_pwr]
        _residue = _residue.subs({self.q:self.q_flq,self.p:self.p_flq},simultaneous=True).subs({self.J:1})
        
        # _residue = (self.residue(order=order).expand()+ sp.O(self.eps**(lead_pwr+1))).removeO()
        # _residue = _residue.subs({self.q:self.q_flq,self.p:self.p_flq},simultaneous=True).subs({self.J:1,self.eps:1})
        
        # Cij found in the expression
        _Cij = _residue.free_symbols - _residue.atoms(sp.Symbol)
        _fun = sp.lambdify((self.phi,*(_Cij)),_residue**2)


        def avg_residue(x):
            Cij = tuple(x)
            return scipy.integrate.quad(lambda phi:_fun(phi,*Cij),0,2*np.pi)[0]

        _values = scipy.optimize.minimize(avg_residue, x0=list(np.zeros(len(_Cij)))).x

        to_sub = {}
        for _val,_C in zip(_values,_Cij):
            to_sub = {**to_sub,**{_C:_val}}
        
        if auto_update:
            self.update(to_sub,verbose=verbose)
        elif verbose:
            display(to_sub)
        if verbose:
            print(60*'-')

        return _values


    def update(self,to_update,verbose=False,deep_update=False):
        if verbose:
            display(to_update)

        for n in range(self.max_order+1):
            # updating the expression
            self.__setattr__(f'K{n}', self.__getattribute__(f'K{n}').subs(to_update, simultaneous=True))

        # Rewrite all variables if needed
        if deep_update:
            for attr in [   'q_map','p_map','q_eps','p_eps','q_flq','p_flq','jacobian',
                            'alpha','beta','mu','gamma']:
                if  self.__getattribute__(attr) is not None:
                    self.__setattr__(attr, self.__getattribute__(attr).subs(to_update, simultaneous=True))

            self.q_iter = sp.lambdify((self.q,self.p),self.q_map)
            self.p_iter = sp.lambdify((self.q,self.p),self.p_map)

            if self._is_poly:
                self.a_values = self.q_map.as_poly(self.q,self.p).as_dict()
                self.b_values = self.p_map.as_poly(self.q,self.p).as_dict()






