import streamlit as st
import pandas as pd
import numpy as np
#lectura de la dimension n de la matriz

st.markdown("METODO GAUSS SEIDEL")

dim_n = int(st.number_input("Escriba la dimension de la matriz (numero de ecuaciones)"))
if dim_n == 0:
    st.stop()

error_esperado = st.number_input("Escriba el error esperado en las soluciones")

df_A = pd.DataFrame()
df_B = pd.DataFrame()

st.write("Matriz A")
for i in range(dim_n):
    df_A["x"+ str(i)] = {0:0}
for i in range(dim_n-1):
    df_A.loc[len(df_A.index)] = 0



edited_df_A = st.experimental_data_editor(df_A)

st.write("Vector B")

df_B["B"] = {0:0}
for i in range(dim_n-1):
    df_B.loc[len(df_B.index)] = 0

edited_df_B = st.experimental_data_editor(df_B)
#lectura de matriz

A = np.array(edited_df_A)
B = np.array(edited_df_B)


#aplicacion del metodo
def seidel(a, x, b):
    # Finding length of a(3)
    n = len(a)
    # itera segun la cantidad de ecuaciones
    for j in range(0, n):
        # temp variable d to store b[j]
        d = b[j]

        # to calculate respective xi, yi, zi
        for i in range(0, n):
            if (j != i):
                d = np.subtract(d, a[j][i] * x[i])
        # updating the value of our solution
        x[j] = d / a[j][j]
    # returning our updated solution
    return x


# int(input())input as number of variable to be solved

# initial solution depending on n(here n=3)
x = [0]*dim_n
x_old = x.copy()

error_arriba_de_umbral = True
errores_x = [1]*dim_n
iteraciones = 0
lista_respuestas = [0]*dim_n
lista_error = [0]*dim_n

# loop run for m times depending on m the error value
while error_arriba_de_umbral and iteraciones<1000:
    x = seidel(A, x, B) #solucion actual
    st.write("Iteracion " + str(iteraciones))
    for i in range(dim_n):
        lista_respuestas[i] = x[i]
    #iterar por cada solucion viendo su error

    if iteraciones==0:
        for i in range(dim_n):
            lista_error[i] = 1
    else:
        error_arriba_de_umbral = False
        for i in range(dim_n): #iterar para todas las respuestas
            lista_error[i] = abs((x[i] - x_old[i]) / x[i]) # sacar el error de cada respuesta
            if lista_error[i] > error_esperado: #si uno de los de la lista de errores tiene mayor al permitido
                error_arriba_de_umbral = True

    data = pd.DataFrame(
        {
            "Soluciones": lista_respuestas,
            "Error" : lista_error
        }
    )
    st.write(data)
    x_old = x.copy()
    iteraciones = iteraciones + 1

#mostrar tabla con las soluciones

