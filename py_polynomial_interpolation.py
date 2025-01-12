#
# TP Métodos Numéricos - 2023
# Alumnos: 
#          • Bianchi, Guillermo
#          • Martin, Denise
#          • Nava, Alejandro

# Profesores: para poder correr adecuadamente el programa es necesario tenes instaladas las
#             bibliotecas de sympy, numpy y matplotlib.
#             Se puede ver el código comentado pero con "play" toda la teoría y práctica aplicada

# Imports
import numpy as np
import sympy as sym
from sympy import *
from sympy import diff
from sympy import lambdify
import random
import ast
from fractions import Fraction
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import mpmath
mpmath.mp.dps = 200
np.seterr(all='ignore', divide='ignore', over='ignore', invalid='ignore')

# Funs
# ------------------------------------------------------------------------------------------------------------
# NEWTON
def my_newton_poly(pares_xy):
    # Se separan los pares en dataset de Y y de X
    xi, yi = separador_pares_x_y(pares_xy)

    # Se calcula el grado del polinomio como <= a n
    n = len(xi)

    x = sym.Symbol('x')

    def newton_poly_by_grade(grade):
        if grade == 0:
            # Para el grado 0, imprime el f(0) correspondiente
            print("Los polinomios por cada par ordenado son:")
            print(f"P_0(x) = {yi[grade]}")
            return yi[grade]
        else:
            # Para grado mayor a cero, crea recursivamente el polinomio anterior en prev_poly
            prev_poly = newton_poly_by_grade(grade - 1)

            # Establece a c como símbolo para expresarlo en la impresión y luego calcularlo
            c = sym.Symbol('c')

            # Selecciona los valores de xi desde x_0 hasta antes del grado actual: "[0, grade)"
            partial_xis = xi[0:grade]

            # Inicializa el nuevo polinomio con la constante c
            new_poly = c

            # Multiplica c por (x - xi) para cada uno de los xi filtrados previamente
            for xi_value in partial_xis: new_poly = new_poly * (x - xi_value)

            # Suma el nuevo polinomio al polinomio de grado anterior
            new_poly = prev_poly + new_poly
            print(f"\nP_{grade}(x) = {new_poly}")
            # print(f"P_{grade}({xi[grade]}) = {new_poly.subs(x, xi[grade])} - ({yi[grade]}) = {new_poly.subs(x, xi[grade]) - yi[grade]}")

            # De dicho polinomio, para el par actual (x[i], y[i], donde "P_grado(x) = f(x)"" ) se despeja la c.
            #       P_grade(x[grade])               = y[grade]
            #       P_grade(x[grade]) - y[grade]    = 0
            solved_c = sym.solve(new_poly.subs(x, xi[grade]) - yi[grade], c)[0]
            print(f"c = {solved_c}")

            # Se reemplaza la C por el valor obtenido resultando en el polinomio de esa iteración
            new_poly = new_poly.subs(c, solved_c)
            print(f"P_{grade}(x) = {new_poly}")

            return new_poly

    poly = newton_poly_by_grade(n-1)
    poly = sym.simplify(poly)

    print("\n\nEl polinomio de Newton obtenido es:")
    print(my_poly_general_format(poly))

    return poly

# ------------------------------------------------------------------------------------------------------------
# LAGRANGE
def my_lagrange_poly(pares):
    # Se separan los pares en dataset de Y y de X
    xi, yi = separador_pares_x_y(pares)

    # Se calcula el grado del polinomio como <= a n
    n = len(xi)

    x = sym.Symbol('x')

    poly = 0

    def g(i):
        tot_mul = 1
        for j in range(n):
            # Son todos los valores de las ordenadas menos el que se evalúa en ese instante
            if i != j:
                # Calcula el producto del cociente entre los valores de x - cada x[i] y la diferencia de ese x[i] - los x[j]
                tot_mul *= (x - xi[j]) / (xi[i] - xi[j])

        return tot_mul

    for i in range(n):
        # Sumatoria de cada polinomio por Xi, dando por resultados el polinomio por Lagrange
        poly += yi[i] * g(i)

    poly = sym.simplify(poly)

    print("El polinomio por Lagrange obtenido es:")
    print(my_poly_general_format(poly))
    return poly

# ------------------------------------------------------------------------------------------------------------
# DIFERENCIAS DIVIDIDAS
def my_divided_diff_poly(pares):
    # Coef es la matriz de datos de my_divided_diff (por diferencias divididas)
    coef = my_divided_diff(pares)
    
    # Se calcula el grado del polinomio como <= a n
    n = len(pares)
    
    # Imprime el paso a paso por cada valor del dataset
    print("Los polinomios por cada par ordenado son:")
    for i in range(n):
        row_str = ""
        for j in range(i + 1):
            if j == 0:
                # Diferencia el primer valor de la lista
                row_str += f"{coef[i, j]}"
            else:
                # Para los demas valores utiliza esto
                # *** Sacada la limitacion de decimales
                row_str += f"{coef[i, j]}(x - {pares[j - 1][0]})"
        print(row_str)

    print("El polinomio por Diferencias Divididas obtenido es:")
    print(my_poly_DD_format(coef))

    return coef

def my_divided_diff(pares):
    # Mide la longitud del data set
    n = len(pares)
    # Se separan los pares en dataset de Y y de X
    x, y = separador_pares_x_y(pares)

    # Crea una matriz del tamaño del dataset y la llena con ceros
    coef = np.zeros([n, n])

    # Establece los valores de Y del set en la primer columna
    coef[:, 0] = y

    # Itera por la segunda columna
    for j in range(1, n):
        for i in range(n - j):
            # Calcula el Coeficiente de la J.tesimaca diferencia dividida
            if x[i + j] == x[i]:
                # Verifica si son iguales los valores para prevenir la division por cero se establece ese coeficiente a cero
                coef[i][j] = 0
            else:
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    # Devuelve la Matriz de Coeficientes        
    return coef

# ------------------------------------------------------------------------------------------------------------
# Generators
def generador_pares(cota_minima, cota_maxima):
    # Genera 20 pares de numeros enteros aleatorios según una cota mínima y máxima
    rango = np.arange(cota_minima, cota_maxima)

    # Para evitar errores de un mismo valor xi con varios yi, el replace=False hace que no peudan repetirse esos 
    # numeros aleatorios. En el caso de yi puede repetirse. Cumpliendo con la Inyectividad
    x_set = np.random.choice(rango, size=20, replace=False)
    y_set = np.random.choice(rango, size=20, replace=True)

    # Ordena los pares de forma ascendente
    lista_pares = list(zip(x_set, y_set))
    return sorted(lista_pares, key=lambda x: x[0])


def separador_pares_x_y(pares):
    print()
    # Establece dos listas vacias para llenar con los valores de y y x
    pares_x = []
    pares_y = []
    # Agrega en cada una los valores
    for i in range(len(pares)):
        pares_x.append(pares[i][0])
        pares_y.append(pares[i][1])
    pares_x = np.array(pares_x)
    pares_y = np.array(pares_y)
    return pares_x, pares_y


def inversor_pares(pares):
    reversed = pares[::-1]
    return reversed


def aleator_pares(pares):
    randomness = pares
    return random.shuffle(randomness)

# ------------------------------------------------------------------------------------------------------------
# Newton funs
def my_newton(poly, x0):
    max_iter = 150
    iteracion = 0
    while iteracion < max_iter:
        # Se establece el error a "e"
        e = 0.00001

        # Se crea la variable
        x = sym.Symbol('x')
        
        funcion = sympify(poly)

        # Se realiza la derivada
        df= funcion.diff(x)
        
        # Verifica que la derivada primera no sea cero, ya que no aseura su convergencia a una raíz
        if df == 0:
            return print("No converge a raiz o lo hara con gran error, ya que la derivada de la funcion es Cero")

        f_x0 = funcion.subs({x: x0}).evalf()
        
        iteracion += 1
            
        if Abs((poly.subs(x,x0)).evalf()) < e:
            return x0
        else:
            return my_newton(poly, x0 - ((poly/df).subs(x, x0).evalf()))    
    raise Exception("No se ha encontrado una raiz en 150 iteraciones")

def my_newton_DD(coef, x_0):
        
    # Castea los coeficientes a Float 128
    coef = np.array(coef, dtype=np.float64)
    
    # Se construye el polinomio completo
    poly = np.poly1d(np.ravel(coef))
        
    # Se establece el error a "e"
    e = 0.00001
    
    # Se toma para iterar la cantidad de coeficientes menos uno
    n = len(coef) - 1
    
    # Se calcula la derivada
    der_coef = [n * coef[i] for i in range(n)]
    der_coef = np.ravel(der_coef)  
    
    # Se construye el polinomio
    derivative_poly = np.poly1d(der_coef)

    root = np.nan

    for i in range(150):
        i += i
        if derivative_poly(x_0) == 0:
            print(f"La derivada en este Punto x = {x_0} es cero y no se puede continuar la busqueda de una raiz")
            return root
        der_val = derivative_poly(x_0)
        if abs(der_val) < e:
            root = x_0
            print(f"El Polinomio por Diferencias Divididas posee una raiz en: ({root:.1f}, 0)        ")
            # Evualuacion de punto en funcion
            result = poly(x)
            print(f"La raiz evaluada en el Polinomio por Diferencias Divididas = ({result})        ")
            return root
        poly_val = poly(x_0)
        if np.isnan(der_val) or np.isnan(poly_val):
            print(f"En el punto x = {x_0} se encuentra un NaN y no se puede continuar la busqueda de una raiz")
            return root
        log_poly_val = np.log(np.abs(poly_val))
        log_der_val = np.log(np.abs(der_val))
        if np.isinf(log_poly_val) or np.isinf(log_der_val):
            print(f"En el punto x = {x_0} se encuentra un infinito y no se puede continuar la busqueda de una raiz")
            return root
        log_x_n = np.log(np.abs(x_0)) - log_poly_val + log_der_val
        if np.isinf(log_x_n):
            print(f"En el punto x = {x_0} se encuentra un infinito y no se puede continuar la busqueda de una raiz")
            return root
        x_n = np.sign(poly_val) * np.exp(log_x_n)
        x_0 = x_n
        if (i == 150):
            # Si no se encuentra un raiz al máximo de iteraciones establecido:
            print("No se ha encontrado una raiz en 150 iteraciones")
            return root

# Bisection fun
def bisection_method(f, a, b, tolerance):
    # verifica el cambio de signo
    if f(a) * f(b) >= 0:
        print("The function does not change sign in the interval [a, b].")
        return None
    
    # realiza iteraciones hasta cumplir con la tolerancia
    while abs(b - a) > tolerance:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

# ------------------------------------------------------------------------------------------------------------        
# Format Print
def my_poly_DD_format(coefs):
    # Toma la primer fila de la matriz de coeficientes
    primer_fila = coefs[0, :]

    # Redondea los valores de la primer_fila a 2 decimales
    # ***** Agrandado rango de decimales a 40
    rounded_fila = np.round_(primer_fila, decimals=40)

    # agrega los valores guardados y genera un polinomio en base a la cantidad de coeficientes
    nice_poly_str = np.poly1d(rounded_fila[::-1])
    
    return nice_poly_str

def my_poly_general_format(poly):
    # Genera un objeto polinómico desde los coefficientes
    coefs = sym.Poly(poly).coeffs()
    n = len(coefs) - 1
    x = sym.symbols('x')
    poly_obj = sym.Poly.from_list(coefs, gens=x)

    # Adquiere el MCM de los denominadores de los coeficientes
    lcm_denom = sym.lcm_list([sym.denom(c) for c in coefs])

    # Multiplica el polinomio por el MCM para obtener los coeficientes
    poly_obj = poly_obj * lcm_denom

    # Convierte el objeto en una expresión SymPy
    expr = poly_obj.as_expr()

    # Agrega el formato string a la expresión
    nice_poly_str = str(expr).replace("**", "^").replace("*", "")

    return nice_poly_str

# ------------------------------------------------------------------------------------------------------------
# Plots
# NEWTON
def graph_details_newton(pares, poly, root, order):
    x, y = separador_pares_x_y(pares)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    x_range = np.linspace(min(x), max(x), 20)

    f = sym.lambdify(sym.Symbol('x'), poly)
    y_range = f(x_range)

    ax.plot(x_range, y_range, color='green')
    
    if not (np.isnan(float(root))):
        root_leyend = 'Punto de raíz'
        plt.plot(root, 0, color='orange', marker='o')
    else:
        root_leyend = 'Raíz no encontrada'
    
    ax.set_title(f"Gráfico de Pares {order} y Polinomio Newton")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    color = ['blue', 'green', 'orange']
    labels = [f'Pares {order}', 'Polinomio Lagrange', f'{root_leyend}']
    handlelist = [plt.plot([], marker="o", ls="", color=color[i])[0] for i in range(3)]
    plt.legend(handlelist, labels, loc='upper left')

    plt.grid(True)
    # plt.legend()
    plt.gca().set_facecolor('#e9edc9')

    plt.show()

# LAGRANGE
def graph_details_lagrange(pares, poly, root, order):
    x, y = separador_pares_x_y(pares)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    x_range = np.linspace(min(x), max(x), 20)

    f = sym.lambdify(sym.Symbol('x'), poly)
    y_range = f(x_range)

    ax.plot(x_range, y_range, color='purple')
    
    if not(np.isnan(float(root))):
        root_leyend = 'Punto de raíz'
        plt.plot(root, 0, color='orange', marker='o')
    else:
        root_leyend = 'Raíz no encontrada'

    ax.set_title(f"Gráfico de Pares {order} y Polinomio Lagrange")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    color = ['blue', 'purple', 'orange']
    labels = [f'Pares {order}', 'Polinomio Lagrange', f'{root_leyend}']
    handlelist = [plt.plot([], marker="o", ls="", color=color[i])[0] for i in range(3)]
    plt.legend(handlelist, labels, loc='upper left')

    plt.grid(True)
    # plt.legend()
    plt.gca().set_facecolor('#e9edc9')

    plt.show()

# DIFERENCIAS DIVIDIDAS
def graph_details_div_diff(pares, poly_coeffs, root, order):
    # Recibe lista de pares y polinomio generado por diferencias divididas en formato String
    x, y = separador_pares_x_y(pares)

    plt.scatter(x, y)
    
    p_callable = my_poly_DD_format(poly_coeffs)

    x_vals = np.linspace(x.min(), x.max(), 20)
    y_vals = p_callable(x_vals)

    plt.ylim(-100, 100)

    plt.plot(x_vals, y_vals, color='red')
    
    if not(np.isnan(float(root))):
        root_leyend = 'Punto de raíz'
        plt.plot(root, 0, color='orange', marker='o')
    else:
        root_leyend = 'Raíz no encontrada'

    plt.title(f"Gráfico de Pares {order} y Polinomio Diferencias Divididas")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    color = ['blue', 'red', 'orange']
    labels = [f'Pares {order}', 'Polinomio Dif. Divididas', f'{root_leyend}']
    handlelist = [plt.plot([], marker="o", ls="", color=color[i])[0] for i in range(3)]
    plt.legend(handlelist, labels, loc='upper left')

    plt.grid(True)
    # plt.legend()
    plt.gca().set_facecolor('#e9edc9')

    plt.show()

# All graphs
def graph_details_all_pairs_n_l(pares, poly_asc, poly_inv, poly_rand, pol_type):
    x, y = separador_pares_x_y(pares)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    x_range = np.linspace(min(x), max(x), 20)
    
    f_asc = sym.lambdify(sym.Symbol('x'), poly_asc)
    y_range_asc = f_asc(x_range)
    ax.plot(x_range, y_range_asc, color='green', label='Polinomio obtenido con pares ordenados', linewidth=6)

    f_inv = sym.lambdify(sym.Symbol('x'), poly_inv)
    y_range_inv = f_inv(x_range)
    ax.plot(x_range, y_range_inv, color='yellow', label='Polinomio obtenido con pares invertidos', linewidth=4)

    f_rand = sym.lambdify(sym.Symbol('x'), poly_rand)
    y_range_rand = f_rand(x_range)
    ax.plot(x_range, y_range_rand, color='red', label='Polinomio obtenido con pares aleatorizados')
    
    ax.set_title(f"Gráfico de Pares y Polinomios obtenidos por {pol_type}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor('#e9edc9')

    plt.show()

def graph_details_all_pairs_DD(pares, poly_asc, poly_inv, poly_rand, pol_type):
    x, y = separador_pares_x_y(pares)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    x_range = np.linspace(min(x), max(x), 20)
    
    f_asc = my_poly_DD_format(poly_asc)
    y_range_asc = f_asc(x_range)
    ax.plot(x_range, y_range_asc, color='green', label='Polinomio obtenido con pares ordenados')

    f_inv = my_poly_DD_format(poly_inv)
    y_range_inv = f_inv(x_range)
    ax.plot(x_range, y_range_inv, color='red', label='Polinomio obtenido con pares invertidos')

    f_rand = my_poly_DD_format(poly_rand)
    y_range_rand = f_rand(x_range)
    ax.plot(x_range, y_range_rand, color='orange', label='Polinomio obtenido con pares aleatorizados')
    
    ax.set_title(f"Gráfico de Pares y Polinomios obtenidos por {pol_type}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor('#e9edc9')

    plt.show()

def graph_details_all(pares, poly_newton, poly_lagrange, poly_coeffs_dd, order):
    x, y = separador_pares_x_y(pares)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    x_range = np.linspace(min(x), max(x), 20)
    
    f_newton = sym.lambdify(sym.Symbol('x'), poly_newton)
    y_range_newton = f_newton(x_range)
    ax.plot(x_range, y_range_newton, color='green', label='Polinomio Newton', linewidth=4)

    f_lagrange = sym.lambdify(sym.Symbol('x'), poly_lagrange)
    y_range_lagrange = f_lagrange(x_range)
    ax.plot(x_range, y_range_lagrange, color='yellow', label='Polinomio Lagrange')

    p_callable = my_poly_DD_format(poly_coeffs_dd)
    y_range_dd = p_callable(x_range)
    ax.plot(x_range, y_range_dd, color='red', label='Polinomio Diferencias Divididas')

    ax.set_title(f"Gráfico de Pares {order} y Polinomios")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor('#e9edc9')

    plt.show()
    

# ------------------------------------------------------------------------------------------------------------
# Prints
#  null) Task + Pres
print("                                                                                  ")
print("**********************************************************************************")
print("*            METODOS NUMERICOS - 2023 - TP METODOs DE INTERPOLACION              *")
print("**********************************************************************************")
print("    ALUMNOS:                                                                      ")
print("            • Bianchi, Guillermo                                                  ")
print("            • Martin, Denise                                                      ")
print("            • Nava, Alejandro                                                     ")
print("                                                                                  ")
print("**********************************************************************************")
print("*                                   OBJETIVO                                     *")
print("**********************************************************************************")
print("  Lograr un polinomio interpolador                                                ")
print("                                                                                  ")
print("**********************************************************************************")
print("*                                   CONSIGNAS                                    *")
print("**********************************************************************************")
print("  Se trabaja con 20 pares de números elegidos al azar, (x, y)                     ")
print("                                                                                  ")
print("  1) Primero establecer cotas para los mismos, generar 20 pares de números con    ")
print("  esas cotas. Ordenar los pares según x.                                          ")
print("  2) Generar un polinomio interpolador por esos 20 pares, según el método de Newton.")
print("   Testear el grado del polinomio obtenido y graficar marcando los pares de datos.")
print("  3) Ordenarlos al revés y obtener otro polinomio interpolador. ¿Qué grado tiene?  ")
print("  ¿Es el mismo? ¿Cómo se hace para saber si es el mismo polinomio? Graficar       ")
print("  4) Lo mismo desordenando los pares.")
print("  5) ¿Se puede poner el programa de obtención de raíces como subrutina de este y  ")
print("  buscar al menos una raíz de uno de los polinomios?   ")
print("  6) Pueden hacer una subrutina que halle por Lagrange, con el mismo conjunto de pares? ")
print("  7) Pueden hacer una subrutina que halle por diferencias divididas, con el       ")
print("  mismo conjunto de pares?                                                        ")
print("  8) ¿Qué se puede decir? ¿Qué conclusiones se pueden sacar?                      ")

#  I) Theory
print("                                                                                  ")
print("**********************************************************************************")
print("*                                      TEORIA                                    *")
print("**********************************************************************************")
print(" Los Métodos de Interpolación son técnicas utilizadas en cálculo para aproximar una")
print(" función desconocida a partir de un conjunto finito de puntos conocidos, hallando ")
print(" un polinomio de grado mínimo que pase por esos puntos                            ")
print("                                                                                  ")
print("                             ********* NEWTON *********                           ")
print(" El método de Newton utiliza un polinomio interpolante que se construye como una  ")
print(" suma de términos de los polinomios conocidos a los que se agrega el producto entre ")
print(" una constante y las raices de los puntos hasta allí utilizados en el polinomio.  ")
print(" Logicamente el primer polinomio es de grado cero, es decir una constante y       ")
print(" Teniendo un polinomio Pm(x) con varios puntos tomados en cuenta, podriamos agregar")
print(" el siguiente punto como: Pm+1(x) = P(x) + c * (x - x0) (x - x1) ... (x - xm)     ")
print(" Donde cada valor de x0, x1 ... xm, representan los valores que anulan al segundo ")
print(" término, para usar el polinomio anterior que se ajustaba a dichos puntos del set")
print(" C es una conmstante que despejada en cada nuevo polinomio, tiene formato:        ")
print("                 ym+1 - Pm(xm+1)                                                  ")
print(" C = ________________________________________                                     ")
print("     (xm+1 - x0) * (xm+1 - x0) *  (xm+1 - x0)                                     ")
print(" Donde ym+1 es el valor correspondiente al par de xm+1, y Pm(xm+1) es el polinomio")
print(" anterior evaluado en el x del set actual: xm+1                                   ")
print(" La cantidad de datos (n) condiciona el grado del polinomio (P):  gr(P) <= n      ")
print("                                                                                  ")
print("                           ********* LAGRANGE *********                           ")
print(" El método de Lagrange utiliza un polinomio interpolante que se construye como una")
print(" suma de polinomios para cada uno de los puntos del set:                          ")
print(" P(x) = L0(x) y0 + L1(x) y1 + … + Ln(x) yn                                        ")
print(" Donde cada polinomio de lagrange ( L ) evaluado en cada punto del set, puede ser ")
print(" expresado genéricamente como:                                                    ")
print("            (x - x0) * (x - x1) ..... ( x - xn )                                  ")
print(" Li(x) = _________________________________________                                ")
print("            (xi - x0) * (xi - x1) .... (xi - xn)                                  ")
print(" En el demonimador ueda logicamente una constante que depende del valor evaluado  ")
print(" y en el numerador las variables que permiten hacer el polinomio.                 ")
print(" La cantidad de datos (n) condiciona el grado del polinomio (P):  gr(P) <= n      ")
print("                                                                                  ")
print("                     ********* DIFERENCIAS DIVIDIDAS *********                    ")
print(" El método de Diferencias Divididas utiliza un polinomio interpolante que se      ")
print(" construye como una sumatoria de diferencias dvididas, según sea:                 ")
print(" • Progresivo:                                                                    ")
print(" Pn−1(x) = f [x0] + f [x0, x1] * (x − x0) + f [x0, x1, x2] * (x − x0) * (x − x1) ···")
print(" .... + f [x0, x1, ... , xn] · (x − x0) · (x − x1) ... (x − xn−1)                 ")
print(" • Regresivo:                                                                    ")
print(" Pn−1(x) = f [xn] + f [xn, xn−1] * (x − xn) + f [xn, xn−1, xn−2] * (x − xn) * (x − xn−1)")
print(" .... + f [xn, xn−1, ... , x1] * (x − xn) · (x − xn−1) ... (x − x1)                ")
print(" Donde se toma todo el dataset y se realiza las diferencias divididas de todos.   ")
print(" En ambos casos el resultado debería ser igual o similar.                         ")
print("                                                                                  ")

#  II) Examples
print("                                                                                  ")
print("**********************************************************************************")
print("*                                    EJEMPLOS                                    *")
print("**********************************************************************************")
print("                                                                                  ")
#________________________________________________________________________________________________
print("                      ********* PARES ASCENDENTES *********                       ")
print(" Se generan los 20 pares de numeros aleatorios enteros.                           ")
pares = generador_pares(-10, 10)
order = "ascendentes"
print("Los 20 pares generados aleatoriamente son:                                        ")
for i in range(len(pares)):
    print(pares[i])
x, y = separador_pares_x_y(pares)
print("                                                                                  ")

print("                             ********* NEWTON *********                           ")
poly_N_asc = my_newton_poly(pares)
print("                                                                                  ")

# Se toma un x0 = 1
root_N = my_newton(poly_N_asc, x0 = 1)
print(f"El Polinomio por Newton posee una raiz en: ({root_N:.2f}, 0)                     ")
# Evualuacion de punto en funcion
result_N = poly_N_asc.subs(sym.Symbol('x'), root_N)
print(f"La raiz evaluada en el Polinomio Newton es = ({result_N})                        ")

graph_details_newton(pares, poly_N_asc, root_N, order)
#my_plot_polynomial_eval(pares, poly_N)
print("                                                                                  ")
print("                           ********* LAGRANGE *********                           ")
poly_L_asc = my_lagrange_poly(pares)
print("                                                                                  ")

# Se toma un x0 = 1
root_L = my_newton(poly_L_asc, x0 = 1 )
print(f"El Polinomio por Lagrange posee una raiz en: ({root_L:.2f}, 0)                   ")
# Evualuacion de punto en funcion
result_L = poly_L_asc.subs(sym.Symbol('x'), root_L)
print(f"La raiz evaluada en el Polinomio Lagrange es = ({result_L})                      ")

graph_details_lagrange(pares, poly_L_asc, root_L, order)
print("                                                                                  ")

print("                     ********* DIFERENCIAS DIVIDIDAS *********                    ")
poly_DD_asc = my_divided_diff_poly(pares)
print("                                                                                  ")

# Se toma un x0 = 1
root_DD = my_newton_DD(poly_DD_asc, x_0 = 1)   

graph_details_div_diff(pares, poly_DD_asc, root_DD, order)
print("                                                                                  ")

# Grafico de las tres funciones superpuestas
graph_details_all(pares, poly_N_asc, poly_L_asc, poly_DD_asc, order)

#________________________________________________________________________________________________
print("                      ********* PARES DESCENDENTES *********                      ")
inversed = inversor_pares(pares)
order = "descendentes"
print("Los elementos invertidos son: \n                                                   ")
for i in range(len(inversed)):
    print(inversed[i])
print("                             ********* NEWTON *********                           ")
poly_N_inv = my_newton_poly(inversed)
print("                                                                                  ")

# Se toma un x0 = 1
root_N = my_newton(poly_N_inv, x0 = 1)
print(f"El Polinomio por Newton posee una raiz en: ({root_N:.2f}, 0)                     ")
# Evualuacion de punto en funcion
result_N = poly_N_inv.subs(sym.Symbol('x'), root_N)
print(f"La raiz evaluada en el Polinomio Newton es = ({result_N})                        ")

graph_details_newton(inversed, poly_N_inv, root_N, order)
print("                                                                                  ")

print("                           ********* LAGRANGE *********                           ")
poly_L_inv = my_lagrange_poly(inversed)
print("                                                                                  ")

# Se toma un x0 = 1
root_L = my_newton(poly_L_inv, x0 = 1 )
print(f"El Polinomio por Lagrange posee una raiz en: ({root_L:.2f}, 0)                   ")
# Evualuacion de punto en funcion
result_L = poly_L_inv.subs(sym.Symbol('x'), root_L)
print(f"La raiz evaluada en el Polinomio Lagrange es = ({result_L})                      ")

graph_details_lagrange(inversed, poly_L_inv, root_L, order)
print("                                                                                  ")

print("                     ********* DIFERENCIAS DIVIDIDAS *********                    ")
poly_DD_inv = my_divided_diff_poly(inversed)
print("                                                                                  ")

# Se toma un x0 = 1
root_DD = my_newton_DD(poly_DD_inv, x_0 = 1)   

graph_details_div_diff(inversed, poly_DD_inv, root_DD, order)
print("                                                                                  ")

# Grafico de las tres funciones superpuestas
graph_details_all(pares, poly_N_inv, poly_L_inv, poly_DD_inv, order)

#________________________________________________________________________________________________
print("                      ********* PARES ALEATORIZADOS *********                     ")
randomness = pares
random.shuffle(randomness)
order = "aleatorizados"
print("Los elementos aleatorizados son: \n                                               ")
for i in range(len(randomness)):
    print(randomness[i])
print("                             ********* NEWTON *********                           ")
poly_N_rand = my_newton_poly(randomness)
print("                                                                                  ")

# Se toma un x0 = 1
root_N = my_newton(poly_N_rand, x0 = 1)
print(f"El Polinomio por Newton posee una raiz en: ({root_N:.2f}, 0)                     ")
# Evualuacion de punto en funcion
result_N = poly_N_rand.subs(sym.Symbol('x'), root_N)
print(f"La raiz evaluada en el Polinomio Newton es = ({result_N})                        ")

graph_details_newton(randomness, poly_N_rand, root_N, order)
print("                                                                                  ")

print("                           ********* LAGRANGE *********                           ")
poly_L_rand = my_lagrange_poly(randomness)
print("                                                                                  ")

# Se toma un x0 = 1
root_L = my_newton(poly_L_rand, x0 = 1 )
print(f"El Polinomio por Lagrange posee una raiz en: ({root_L:.2f}, 0)                   ")
# Evualuacion de punto en funcion
result_L = poly_L_rand.subs(sym.Symbol('x'), root_L)
print(f"La raiz evaluada en el Polinomio Lagrange es = ({result_L})                      ")

graph_details_lagrange(randomness, poly_L_rand, root_L, order)
print("                                                                                  ")

print("                     ********* DIFERENCIAS DIVIDIDAS *********                    ")
poly_DD_rand = my_divided_diff_poly(randomness)
print("                                                                                  ")

# Se toma un x0 = 1
root_DD = my_newton_DD(poly_DD_rand, x_0 = 1)   
graph_details_div_diff(randomness, poly_DD_rand, root_DD, order)
print("                                                                                  ")

# Grafico de las tres funciones superpuestas
graph_details_all(pares, poly_N_rand, poly_L_rand, poly_DD_rand, order)

# Graficos de las 3 funciones obtenidas segun orden
graph_details_all_pairs_n_l(pares, poly_N_asc, poly_N_inv, poly_N_rand, "Newton")
graph_details_all_pairs_n_l(pares, poly_L_asc, poly_L_inv, poly_L_rand, "Lagrange")
graph_details_all_pairs_DD(pares, poly_DD_asc, poly_DD_inv, poly_DD_rand, "Diferencias Divididas")

## IV) Conclusions
print("                                                                                  ")
print("**********************************************************************************")
print("*                                  CONCLUSIONES                                  *")
print("**********************************************************************************")
print(" • El método de Lagrange utiliza una fórmula de interpolación que se basa en las  ")
print("   ordenadas de una función en cambio Newton y Diferencias Divididas requieren las")
print("   ordenadas y absisas para su desarrollo.                                        ")
print("                                                                                  ")
print(" • El método de Lagrange solo se utiliza para la interpolación, mientras que el de")
print("   Newton y Diferencias Divididas se pueden utilizar para interpolación y extrapolación.")
print("                                                                                  ")
print(" • La ventaja del método de Newton es que es idea para datos en gran escala.       ")
print("   Sin embargo, requiere calcular un nuevo polinomio con cada par ingresado.     ")
print("                                                                                  ")
print(" • La ventaja del método de Lagrange es que es fácil de entender y aplicar.       ")
print("   Sin embargo, no es eficiente para grandes conjuntos de datos (limitación).     ")
print("                                                                                  ")
print(" • La ventaja del método de Diferencias Divididas es que es sencillo y no hay que ")
print("   recalcular coeficientes. Sin embargo, tiene que tener suficiente presición para")
print("   entenderse y no permite cambios muy grandes (por su similitud con derivadas).  ")
print("                                                                                  ")
print(" • Los tres métodos permiten encontrar una función polinómica que pase por un     ")
print("   conjunto de puntos de manera continua, generalizando la propiedad euclidiana de")
print("   que por dos puntos distintos pasa siempre una (única) recta.                   ")
print("                                                                                  ")
print(" • Respecto a la exactitud de los polinomios, es observable la diferencia entre   ")
print("   Lagrange y Newton que interpolan polinomios iguales y mas cercanos a los pares ")
print("   ordenados originales, y el de diferencias divididas cuya exactitud es menor    ")
print("   y sólo concuerda algunos pares ordenados originales.                           ")
print("                                                                                  ")
print(" • En cuanto a los grados de los polinomios encontrados, podemos observar que todos")
print("   cumplen con Gr(p) < n, y se obtienen en los distintos casos, diferentes grados ")
print("   pero siempre en cumplimiento de esa regla.                                     ")
print("                                                                                  ")
print(" • Las raices de los polinomios se obtuvieron por Newton-Raphson, en la mayoria de")
print("   los casos se da una convergencia a una raiz, pero puede suceder que dado el valor")
print("   inicial ingresado, no halle ninguna.                                           ")
print("   Por otra parte también peude verse casos donde la raiz esta fuera del rango de ")
print("   los pares ordenados provistos y es visible graficamente como coordenada, fuera ")
print("   de la curva del polinomio hallado.                                             ")
print("                                                                                  ")
print(" • En las gráficas y cálculos puede observarse como los polinomios de Newton y    ")
print("   Lagrange, se mantienen constantes a pesar del cambio de los pares ordenados a: ")
print("   ascendente, descendente y aletorio. Sin embargo, en el caso del polinomio de   ")
print("   diferencias divididas, hay leves cambios en los cálculos y por ende las gráficas")
print("   difieren levemente, esto se debe a la relación entre los partes ordenados que  ")
print("   en el procesamiento de este polinomio.                                         ")
print("   En el caso de Lagrange se mantiene ya que realiza en un solo cálculo con todos ")
print("   los pares ordenados. Y para Newton, si bien la convergencia al polinomio modifica")
print("   los pasos intermedios, el cálculo final depende exclusivamente de los valores   ")
print("   de x, por lo que en todos los casos terminará con el mismo resultado.           ")
print("                                                                                  ")
print(" • Cabe destacar que existe el fenómeno Runge como problema de oscilación en los   ")
print("   extremos de un intervalo. Esto implica que el polinomio interpolado, puede     ")
print("   ajustarse idealmente a los valores intermedios pero no en los extremos.        ")
print("   Aquí se puede observar el comportamiento en las gráficas de Diferencias        ")
print("   Divididas, donde teniendo puntos equidistantes con abruptas subidas y bajadas, ")
print("   además de su parecido a la derivación, genera dicho fenómeno.                  ")
print("                                                                                  ")
print(" • NOTA1: en las líneas 178 y 179 se encuentra el generador de pares, donde dice: ")
print("   size, se puede modificar el valor para visualizar mejor y con menos cambios abruptos")
print("   la interpolación de los polinomios, por ejemplo en 3 podrán verse las funciones")
print("   cuadráticas con curvas mas suaves y más probabilidad de encontrar una raíz.    ")
print("                                                                                  ")
print(" • NOTA2: es ideal correr el programas más de una vez para poder apreciar la diferencia")
print("   en grados de los polinomios y las distintas gráficas de cada uno.")
print("                                                                                  ")
print(" • NOTA3: en la misma función de la NOTA1, puede verse la cota máxima y mínima como ")
print("   rangos de los pares ordenados, en este caso decidimos realizarlo de -10 a 10  ")
print("   para asegurar una raíz, sin embargo como se anotó previamente, algunas raíces  ")
print("   se pueden observar en el gráfico como 'continuación teórica' del polinomio en ")
print("   cuestión; aunque exceda el rango dado en los pares creados.")
print("                                                                                  ")
print(" • NOTA4: si bien se trató de considerar todos los posibles errores de desborde en")
print("   Newton, en especial para diferencias divididas, puede observarse algún error   ")
print("   antes de la impresión del error contemplado. Para evitarlo es posible normalizar")
print("   los valores, sin embargo esto interferiría con las comparaciones de los otros dos")
print("   polinomios, por lo que decidimos despreciar esa línea de error ya que el motivo")
print("   de la misma se justifica en la siguiente impresión.                            ")
print("                                                                                  ")
