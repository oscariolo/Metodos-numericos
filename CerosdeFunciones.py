import math as mt
import pandas as pd
import streamlit as st
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import AxesZero

def Biseccion(F,error_buscado):
    st.header("METODO DE BISECCION")

    x0 = st.number_input("Dominio 1")

    x1 = st.number_input("Dominio 2")

    x2 = 0

    error = 1

    iteraciones = 0

    aux = 0

      # Error que se desea obtener
    x0list = [x0]
    x1list = [x1]
    x2list = [x2]
    fx0list = [F.subs(x, x0)]
    fx1list = [F.subs(x, x1)]
    fx2list = [F.subs(x, x2)]
    errorlist = [error]

    try:
        while (error > error_buscado and F.subs(x, x2) != 0):
            x2 = (x1 + x0) / 2
            if (F.subs(x, x0) * F.subs(x, x2)) < 0:
                x1 = x2
            else:
                x0 = x2

            if iteraciones == 0: #primera iteracion se evita dividir para 0
                aux = x2
            else:
                error = abs((x2 - aux) / x2) * 100

                aux = x2

            iteraciones += 1
            x0list.append(x0)
            x1list.append(x1)
            x2list.append(x2)
            fx0list.append(F.subs(x, x0))
            fx1list.append(F.subs(x, x1))
            fx2list.append(F.subs(x, x2))
            errorlist.append(error)

        data = pd.DataFrame(
            {
                "iteraciones": range(0, iteraciones + 1),
                "x0": x0list,
                "x1": x1list,
                "x2": x2list,
                "f(x0)": fx0list,
                "f(x1)": fx1list,
                "f(x2)": fx2list,
                "Error": errorlist

            }

        )

        xs = np.linspace(-10, 10, 100)
        ys = [sy.lambdify(x, F)(x_val) for x_val in xs]
        fig = plt.figure()
        ax = fig.add_subplot(axes_class=AxesZero)
        ax.plot(xs, ys)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.set_title('Gráfica de la función')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        st.pyplot(fig)


        st.write(data)
    except TypeError:
        st.write("Funcion inválida o de dominio")
def Punto_Fijo(F,error_buscado):
    st.header("METODO DE PUNTO FIJO")

    try:

        if (string_function == ""):
            st.stop()
        else:
            recommend_g = string_function + "+ x"
            st.write("Function recomendada: \n" + str(sy.sympify(recommend_g)))
            g_function_input = st.text_input("Escriba la funcion g(x)", key="2")

        G = sy.sympify(g_function_input)

        x0 = st.number_input("Escriba el punto")

        # por umbral
        error = 1
        iteraciones = 0
        x2 = 0
        aux = 0
        errorlista = [error]
        x2lista = [x2]
        resultslista = [F.subs(x, x2)]

        while (error > error_buscado and iteraciones < 50):
            x2 = float(G.subs(x, x0))  # actual

            error = abs((x2 - x0) / (x2)) * 100
            x0 = x2

            errorlista.append(error)
            x2lista.append(x2)
            resultslista.append(F.subs(x, x2))
            iteraciones += 1

        data = pd.DataFrame(
            {
                "iteraciones": range(0, iteraciones + 1),
                "x2": x2lista,
                "f(x2)": resultslista,
                "error": errorlista

            }

        )

        st.write(data)
    except TypeError:
        st.write("Funcion inválida o de dominio")

def Newton_Rhapson(F,error_esperado):
    st.header("METODO DE NEWTON RHAPSON")

    Fd = sy.diff(F, x)
    x2 = 0
    iteraciones = 0

    x0 = st.number_input("Punto de inicio")
    error = float(1)
    error_esperado = 0.05

    x2lista = [x0]
    resultslista = [F.subs(x, x0)]
    errorlista = [error]

    # por umbral
    try:
        while (error > error_esperado and iteraciones < 30):
            x2 = x0 - F.subs(x, x0) / Fd.subs(x, x0)
            if iteraciones == 0:
                x0 = x2
            else:

                error = abs((x2 - x0) / (x2)) * 100
                x0 = x2

            iteraciones += 1

            errorlista.append(error)
            x2lista.append(x2)
            resultslista.append(F.subs(x, x2))

        data = pd.DataFrame(
            {
                "iteraciones": range(0, iteraciones + 1),
                "x2": x2lista,
                "f(x2)": resultslista,
                "error": errorlista

            }

        )

        st.write(data)
    except TypeError:
        st.write("Funcion inválida o de dominio")
    # por iteraciones


def Secante(F,error_esperado):
    st.header("METODO DE SECANTE")

    x0 = st.number_input("Punto x0")
    x1 = st.number_input("Punto x1")
    x2 = 0

    error = 1
    iteraciones = 0

    x2lista = [x2]
    listaerror = [error]
    resultslista = [F.subs(x, x1)]

    # por error
    try:
        while (error > error_esperado and iteraciones < 30):

            x2 = x1 - F.subs(x, x1) * (x0 - x1) / (F.subs(x, x0) - F.subs(x, x1))

            if iteraciones == 0:
                x0 = x1
                x1 = x2
            else:
                error = abs((x2 - x1) / (x2)) * 100
                x0 = x1
                x1 = x2

            iteraciones += 1

            listaerror.append(error)
            x2lista.append(x2)
            resultslista.append(F.subs(x, x2))

        data = pd.DataFrame(
            {
                "iteraciones": range(0, iteraciones + 1),
                "x2": x2lista,
                "f(x2)": resultslista,
                "error": listaerror

            }

        )

        st.write(data)
    except TypeError:
        st.write("Funcion inválida o dominio")

def Muller(F,error_esperado):
    st.header("METODO DE MULLER")
    x = sy.symbols("x")

    a = st.number_input("Punto x1",key="a")
    b = st.number_input("Punto x2",key="b")
    c = st.number_input("Punto x3",key="c")

    res = 0
    error = 1
    iteraciones = 0

    x2lista = [c]
    listaerror = [error]
    resultslista = [F.subs(x, c)]

    MAX_ITERATIONS = 10000
    try:
        while error > error_esperado and iteraciones < MAX_ITERATIONS:
            # Calculating various constants
            # required to calculate x3
            f1 = (F.subs(x, a))
            f2 = (F.subs(x, b))
            f3 = (F.subs(x, c))

            d1 = f1 - f3
            d2 = f2 - f3
            h1 = a - c
            h2 = b - c
            a0 = f3
            a1 = (((d2 * mt.pow(h1, 2)) -
                   (d1 * mt.pow(h2, 2))) /
                  ((h1 * h2) * (h1 - h2)))
            a2 = (((d1 * h2) - (d2 * h1)) /
                  ((h1 * h2) * (h1 - h2)))
            x_ = ((-2 * a0) / (a1 +
                              abs(mt.sqrt(a1 * a1 - 4 * a0 * a2))))
            y_ = ((-2 * a0) / (a1 -
                              abs(mt.sqrt(a1 * a1 - 4 * a0 * a2))))

            # Taking the root which is
            # closer to x2
            if x_ >= y_:
                res = x_ + c
            else:
                res = y_ + c

            a = b
            b = c
            c = res

            error = abs((c - b) / (c)) * 100

            listaerror.append(error)
            x2lista.append(c)
            resultslista.append(F.subs(x, c))

            iteraciones += 1

        data = pd.DataFrame(
            {
                "iteraciones": range(0, iteraciones + 1),
                "x2": x2lista,
                "f(x2)": resultslista,
                "error": listaerror

            }

        )

        st.write(data)
        st.write("El valor del cero de funcion encontrado es",round(res, 4))
    except TypeError:
        st.write("Dominio o función inválida")

st.title("METODOS PARA RAIZ DE FUNCIONES")

f_pfijo = st.checkbox("PUNTO FIJO")
f_biseccion = st.checkbox("BISECCION")
f_NewtonR = st.checkbox("NEWTON-RHAPSON")
f_Secante = st.checkbox("SECANTE")
f_Muller = st.checkbox("MULLER")

string_function = st.text_input("Escriba la funcion f(x)=")


error_buscado = st.number_input("Escriba el error que desea tener su aproximación (error relativo)")

x = sy.symbols("x")
F = sy.sympify(string_function)


#######COMPROBACIONES##########

#comprobar convergencia lenta
def slowConvergence(error, errorₚᵣₑᵥ, slow_convergence_counter, cadena_tip):
    convergencia = 0.009

    convergencia_lenta = round(abs((error - errorₚᵣₑᵥ) / error), digits=5)

    if (convergencia_lenta <= convergencia):
        slow_convergence_counter += 1
        if (slow_convergence_counter == 3):
            st.write("CONVERGENCIA LENTA\n$cadena_tip")
            slow_convergence_counter = -1
    else:
        slow_convergence_counter = 0
    return slow_convergence_counter

    # Comprobar contiuidad de una funcion

def isContinuos(f, a, b, ϵ):
    n = 100 / ϵ
    i=1
    for i in range(n - 1):
        x1 = a + (i - 1) * (b - a) / (n - 1)
        x2 = a + i * (b - a) / (n - 1)

    if mt.isclose(f.subs(x,x1), f.subs(x,x2), atol=ϵ, rtol=ϵ):
        st.write("La funcion no es continua en el intervalo [$a, $b]")
        return False
    return True
#oscilante
#Comprobar oscilacion

def oscilacion(range_size,range_sizeₚᵣₑᵥ,oscillation_counter,cadena_tip):
    if range_size == range_sizeₚᵣₑᵥ:
        oscillation_counter += 1
        if oscillation_counter==5:
            st.write("Oscilacion" + str(cadena_tip))
            oscillation_counter =- 1
    else:
        oscillation_counter=0
    return oscillation_counter

#Comprobar si es real

    # Comprobar funcion real
def isRealFunction(f, a, b, ϵ, tolerancia):
    n = 100 / ϵ
    incremento = abs(b - a) / n

    while a < b:
        if not mt.isclose(f.subs(x,a.imag), 0, atol=tolerancia, rtol=tolerancia):
            st.write("La funcion no es real en el intervalo proporcionado")
            return False
        a += incremento
    return True

###############################


if f_pfijo:
    Punto_Fijo(F,error_buscado)
#

if f_biseccion:
    Biseccion(F,error_buscado)

if f_NewtonR:
    Newton_Rhapson(F,error_buscado)

if f_Secante:
    Secante(F,error_buscado)

if f_Muller:
    Muller(F,error_buscado)
