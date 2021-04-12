from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Dense
transformador = StandardScaler()


def criar_classes(dados):
    for classe in dados.columns:
        dados[classe] = LabelEncoder().fit_transform(dados[classe])

    return dados


def repartir_dados(dados):
    entradas = dados.iloc[:, 1:]
    resultados = dados.iloc[:, :1]
    return entradas, resultados


def deletar_dados_desnecessários(dados):
    del dados['veil-type']
    return dados


def preparar_dados(dados):
    dados = deletar_dados_desnecessários(dados)
    dados.dropna(inplace=True)
    dados = criar_classes(dados)
    entradas, resultados = repartir_dados(dados)
    entradas = transformador.fit_transform(entradas)
    return train_test_split(entradas, resultados, test_size=0.2)


if __name__ == '__main__':

    dados = read_csv('dados.csv')
    entradas_treino, entradas_teste, resultados_treino, resultados_teste = preparar_dados(dados)

    modelo = Sequential()
    modelo.add(Dense(10, activation="relu", input_shape=(entradas_treino.shape[1], )))
    modelo.add(Dense(1, activation="sigmoid"))

    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    modelo.fit(entradas_treino, resultados_treino, epochs=500)

    score = modelo.evaluate(entradas_teste, resultados_teste)
    assertividade = "%.2f" % ((-score[0] + score[1]) * 100)

    print(f"assertividade: {assertividade}%")