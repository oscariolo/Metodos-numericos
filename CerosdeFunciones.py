import math
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

    isContinuos(F,x0,x1)

    x2 = 0

    error = 1

    iteraciones = 0

    aux = 0
    isReal(F,x0)
      # Error que se desea obtener
    x0list = [x0]
    x1list = [x1]
    x2list = [x2]
    fx0list = [F.subs(x, x0)]
    fx1list = [F.subs(x, x1)]
    fx2list = [F.subs(x, x2)]
    errorlist = [error]
    isReal(F, x1)
    try:
        while (error > error_buscado and F.subs(x, x2) != 0):
            x2 = (x1 + x0) / 2
            isReal(F, x2)
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
        xs = np.linspace(-10, 10, 100)
        #definimos cada iteracion una ecuacion de la recta
        #pendiente igual a valor evaluado en diff

        xpoints = np.linspace(-10, 10, 100)
        fig, ax = plt.subplots()



        for i in range(iteraciones): #segun el numero de iteraciones
            m = Fd.subs(x, x2lista[i])
            #ecuacion de la recta
            #arrays para esa recta
            y = [None]*xpoints.size
            for j in range(xpoints.size):
                y[j] = m *(xpoints[j] - x2lista[i]) + F.subs(x,resultslista[i])

            ax.plot(xpoints,y, label = "Iteracion ")

        for j in range(xpoints.size):
            y[j] = F.subs(x,xpoints[j])


        ax.plot(xpoints,y,label = "Funcion")
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

        plt.plot(x2, sy.lambdify(x, F)(x2), 'ro', label='Raíz')
        st.pyplot(fig)
        st.write(data)
    except TimeoutError:
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

        oscilacion(np.array(x2lista))

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

def isContinuos(f, a, b):
    es_continua = True

    for i in range(int(a), int(b) + 1):
        limite_izquierdo = f.limit(x, i, dir='-')
        limite_derecho = f.limit(x, i, dir='+')
        valor_funcion = f.subs(x, i)

        if limite_izquierdo != limite_derecho or limite_izquierdo != valor_funcion:
            es_continua = False
            break

    if es_continua:
        st.write("La función es continua en el intervalo [{}, {}]".format(a, b))

    else:
        st.write("La función no es continua en el intervalo [{}, {}]".format(a, b))
        st.stop()
#oscilante
#Comprobar oscilacion

def oscilacion(array):
    respuestas = array

    # Verificar si las respuestas oscilan
    oscila = False

    for i in range(len(respuestas) - 1):
        if np.sign(respuestas[i]) != np.sign(respuestas[i + 1]) and np.abs(respuestas[i + 1] - respuestas[i]) > 1:
            oscila = True
            break

    if oscila:
        st.write("Las respuestas oscilan.")
        st.stop()

def isReal(f,x0):
    function = sy.lambdify(x, f)
    resultado_obtenido = function(x0)
    if  math.isnan(resultado_obtenido):
        st.write("La función f(x) no es real en el intervalo dado")
        st.stop()
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
