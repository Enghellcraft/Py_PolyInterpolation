# TP Métodos Numéricos - 2023
# Alumnos: 
#          • Bianchi, Guillermo
#          • Martin, Denise
#          • Nava, Alejandro

# Imports
import numpy as np
import sympy as sym
import random
from fractions import Fraction
import matplotlib.pyplot as plt

# Funs
# ------------------------------------------------------------------------------------------------------------
# NEWTON
def my_newton_poly(pares_xy):
    # Se separan los pares en dataset de Y y de X
    xi, yi = separador_pares_x_y(pares_xy)

    # Se calcula el grado del polinomio como <= a n
    n = len(xi)

    x = sym.Symbol('x')

    def newton_poly_for_grade(grade):
        if grade == 0:
            # Para el grado 0, imprime el Y[0] correspondiente
            print("Los polinomios por cada par ordenado son:")
            print(f"P_0(x) = {yi[grade]}")
            return yi[grade]
        else:
            # Para grado mayor a cero, crea el polinomio anterior en prev_poly
            prev_poly = newton_poly_for_grade(grade - 1)

            # Establece a c como símbolo para expresarlo en la impresión y luego calcularlo
            c = sym.Symbol('c')

            # Selecciona los valores de xi con el i (grado) correspondiente
            partial_xis = xi[0:grade]

            # Inicializa el nuevo polinomio con la constante c
            new_poly = c

            # Multiplica todos los (x - xi) para cada xi 
            for xi_value in partial_xis: new_poly = new_poly * (x - xi_value)

            # Combina el polinomio anterior con el nuevo polinomio
            new_poly = prev_poly + new_poly
            print(f"\nP_{grade}(x) = {new_poly}")
            # print(f"P_{grade}({xi[grade]}) = {new_poly.subs(x, xi[grade])} - ({yi[grade]}) = {new_poly.subs(x, xi[grade]) - yi[grade]}")

            # Del polinomio anterior evaluado en el actual x[i] y el y[i], se despeja la c
            solved_c = sym.solve(new_poly.subs(x, xi[grade]) - yi[grade], c)[0]
            print(f"c = {solved_c}")

            # se reemplaza la C por el valor obtenido resultando en el polinomio de esa iteración
            new_poly = new_poly.subs(c, solved_c)
            print(f"P_{grade}(x) = {new_poly}")

            return new_poly

    poly = newton_poly_for_grade(n-1)
    poly = sym.simplify(poly)

    print("\n\nEl polinomio de Newton obtenido es:")
    print(poly)

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
    # poly = sym.cancel(poly)
    # poly = sym.factor(poly)

    print("El polinomio por Lagrange obtenido es:")
    print(poly)
    return poly

# ------------------------------------------------------------------------------------------------------------
# DIFERENCIAS DIVIDIDAS
def my_divided_diff_poly(pares):
    # Coef es la matriz de datos de my_divided_diff (por diferencias divididas)
    coef = my_divided_diff(pares)
    n = len(pares)

    # Toma la primer fila de la matriz de coeficientes de diferencias divididas
    primer_fila = coef[0, :]

    # Redondea los valores de la primer_fila a 2 decimales
    rounded_fila = np.round_(primer_fila, decimals=2)

    # agrega los valores guardados y genera un polinomio en base a la cantidad de coeficientes
    newton_poly_str = np.poly1d(rounded_fila[::-1])

    # Imprime el paso a paso por cada valor del dataset
    print("Los polinomios por cada par ordenado son:")
    for i in range(n):
        row_str = ""
        for j in range(i + 1):
            if j == 0:
                # Diferencia el primer valor de la lista
                row_str += f"{coef[i, j]:+.2f}"
            else:
                # Para los demas valores utiliza esto
                row_str += f"{coef[i, j]:+.2f}(x - {pares[j - 1][0]})"
        print(row_str)

    print("El polinomio por Diferencias Divididad obtenido es:")
    print(newton_poly_str)

    return newton_poly_str

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
                # Verifica si son iguales los valores para prevenir la division por cero
                # establece ese coeficiente a cero
                coef[i][j] = 0
            else:
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    # Devuelve la Matriz de Coeficientes        
    return coef


# ------------------------------------------------------------------------------------------------------------
# Generators
def generador_pares(cota_minima, cota_maxima):
    # Genera 20 pares de numeros enteros aleatorios según una cota mínima y máxima
    rango = np.arange(cota_minima, cota_maxima + 1)

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

def generador_de_raices(poly):
    x = sym.Symbol('x')
    coeff = sym.Poly(poly, x).all_coeffs()
    roots = np.roots(coeff)
    print ("Las Raices del Polinomio son:")
    for i, root in enumerate(roots):
        print (f"Raíz {i+1} =  {root:.2f}")

# ------------------------------------------------------------------------------------------------------------
# Plots
# NEWTON
def graph_details_newton(pares, poly):
    x, y = separador_pares_x_y(pares)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    x_range = np.linspace(min(x), max(x), 5)

    f = sym.lambdify(sym.Symbol('x'), poly)
    y_range = f(x_range)

    ax.plot(x_range, y_range, color='green')

    ax.set_title("Gráfico de Pares ordenados y Polinomio Newton")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.grid(True)
    # plt.legend()
    plt.gca().set_facecolor('#e9edc9')

    plt.show()

# LAGRANGE
def graph_details_lagrange(pares, poly):
    x, y = separador_pares_x_y(pares)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    x_range = np.linspace(min(x), max(x), 5)

    f = sym.lambdify(sym.Symbol('x'), poly)
    y_range = f(x_range)

    ax.plot(x_range, y_range, color='blue')

    ax.set_title("Gráfico de Pares ordenados y Polinomio Lagrange")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.grid(True)
    # plt.legend()
    plt.gca().set_facecolor('#e9edc9')

    plt.show()

# DIFERENCIAS DIVIDIDAS
def graph_details_div_diff(pares, poly_str):
    # Recibe lista de pares y polinomio generado por diferencias divididas en formato String
    x, y = separador_pares_x_y(pares)

    plt.scatter(x, y)

    # Declaro el simbolo como x
    x = sym.Symbol('x')

    # Conversion de polinomio string a simbolo
    poly_expr = sym.sympify(poly_str)
    poly_func = sym.lambdify(x, poly_expr)

    x_vals = np.linspace(x.min(), x.max(), 5)
    y_vals = poly_func(x_vals)

    plt.plot(x_vals, y_vals, color='red')

    plt.title("Gráfico de Pares ordenados y Polinomio Diferencias Divididas")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.grid(True)
    # plt.legend()
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
print("  ¿Es el mismo? ¿Cómo se hace para saber si el el mismo polinomio? Graficar       ")
print("  4) Lo mismo desordenando los pares.")
print("  5) ¿Se puede poner el programa de obtención de raíces como subrutina de este y  ")
print("  buscar al menos una raíz de uno de los polinomios?   ")
print("  6) Pueden hacer una subrutina que halle por Lagrange, con el mismo conjunto de pares? ")
print("  7) Pueden hacer una subrutina que halle por diferencias divididas, con el       ")
print("  mismo conjunto de pares?                                                        ")
print("  ¿Qué se puede decir? ¿Qué conclusiones se pueden sacar?                         ")

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
print(" Se generan los 20 pares de numeros aleatorios enteros.                           ")
pares = generador_pares(0, 20)
print("Los 20 pares generados aleatoriamente son:                                        ")
for i in range(len(pares)):
    print(pares[i])
x, y = separador_pares_x_y(pares)
print("                                                                                  ")

print("                             ********* NEWTON *********                           ")

poly_N = my_newton_poly(pares)
graph_details_newton(pares, poly_N)
print("                                                                                  ")
generador_de_raices(poly_N)
print("                                                                                  ")
print("                           ********* LAGRANGE *********                           ")
poly_L = my_lagrange_poly(pares)
graph_details_lagrange(pares, poly_L)
print("                                                                                  ")
generador_de_raices(poly_L)
print("                                                                                  ")
print("                     ********* DIFERENCIAS DIVIDIDAS *********                    ")
poly_DD = my_divided_diff_poly(pares)
graph_details_div_diff(pares, poly_DD)
print("                                                                                  ")
generador_de_raices(poly_DD)
print("                                                                                  ")
inversed = inversor_pares(pares)
print("Los elementos invertidos son: \n                                                   ")
for i in range(len(inversed)):
    print(inversed[i])

randomness = pares
random.shuffle(randomness)
print("Los elementos aleatorizados son: \n                                               ")
for i in range(len(randomness)):
    print(randomness[i])

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
print("                                                                                  ")
