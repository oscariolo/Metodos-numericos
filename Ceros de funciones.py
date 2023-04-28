import math as mt

import pandas as pd
import streamlit as st
import sympy as sy


def Biseccion(F):
    st.header("METODO DE BISECCION")

    # if F == NaN;
    #   st.stop()
    #
    # x₀=x0
    x0 = st.number_input("Dominio 1")

    # x₁=x1
    x1 = st.number_input("Dominio 2")

    # x₂=0  # PROCESO ITERATIVO-VALOR PROXIMO PARA CERO EN FUNCION

    x2 = 0

    error = 1  # INICIALIZACION DEL ERROR

    iteraciones = 0  # Numero de iteraciones

    aux = 0  # Valor auxiliar

    error_buscado = 0.005  # Error que se desea obtener
    x0list = [x0]
    x1list = [x1]
    x2list = [x2]
    fx0list = [F.subs(x, x0)]
    fx1list = [F.subs(x, x1)]
    fx2list = [F.subs(x, x2)]
    fx0x2list = [F.subs(x, x0) * F.subs(x, x2)]
    errorlist = [error]

    while (error > error_buscado and F.subs(x, x2) != 0):
        x2 = (x1 + x0) / 2
        if (F.subs(x, x0) * F.subs(x, x2)) < 0:
            x1 = x2
        else:
            x0 = x2

        if iteraciones == 0:
            aux = x2
        else:
            error = abs((x2 - x0) / (x2)) * 100
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

    st.write(data)
    st.write("Valor de x mas cercano calculado: \n x2=  " + str(x2) + " \n f(x2)= " + str(F.subs(x, x2)))


def Punto_Fijo(F):
    st.header("METODO DE PUNTO FIJO")

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
    error_buscado = 0.05
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


def Newton_Rhapson(F):
    st.header("METODO DE NEWTON RHAPSON")

    Fd = sy.diff(F, x)
    x2 = 0
    iteraciones = 0

    x0 = st.number_input("Punto de inicio")
    error = 1
    error_esperado = 0.05

    x2lista = [x0]
    resultslista = [F.subs(x, x0)]
    errorlista = [error]

    # por umbral
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

    # por iteraciones


def Secante(F):
    st.header("METODO DE SECANTE")

    x0 = st.number_input("Punto x0")
    x1 = st.number_input("Punto x1")
    x2 = 0

    error = 1
    error_esperado = 0.05
    iteraciones = 0

    x2lista = [x2]
    listaerror = [error]
    resultslista = [F.subs(x, x1)]

    # por error

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


def Muller(F):
    st.header("METODO DE MULLER")
    x = sy.symbols("x")

    a = st.number_input("Punto x1",key="a")
    b = st.number_input("Punto x2",key="b")
    c = st.number_input("Punto x3",key="c")

    res = 0
    error = 1
    error_esperado = 0.05
    iteraciones = 0

    x2lista = [c]
    listaerror = [error]
    resultslista = [F.subs(x, c)]

    MAX_ITERATIONS = 10000

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


st.title("METODOS PARA RAIZ DE FUNCIONES")

f_pfijo = st.checkbox("PUNTO FIJO")
f_biseccion = st.checkbox("BISECCION")
f_NewtonR = st.checkbox("NEWTON-RHAPSON")
f_Secante = st.checkbox("SECANTE")
f_Muller = st.checkbox("MULLER")

string_function = st.text_input("Escriba la funcion f(x)=")
x = sy.symbols("x")
F = sy.sympify(string_function)

if f_pfijo:
    Punto_Fijo(F)
#

if f_biseccion:
    Biseccion(F)

if f_NewtonR:
    Newton_Rhapson(F)

if f_Secante:
    Secante(F)

if f_Muller:
    Muller(F)
