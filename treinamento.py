# Importar a biblioteca
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import pandas as pd 
import joblib 

# Importação de dados
url = r'C:\Users\Aluno\Desktop\Projeto_Cardiaco_william\previsao_doenca_cardiaca.csv.csv' # Define a URL do arquivo CSV 
dados_cardiaco = pd.read_csv(url, sep=',') # Lê o arquivo CSV e armazena em um DataFrame pandas

# Renomeando colunas
nova_coluna_nomes = { # Dicionário com os novos nomes das colunas
    'Chest_Pain': 'Dor_no_Peito',
    'Shortness_of_Breath': 'Falta_de_Ar',
    'Fatigue': 'Fadiga',
    'Palpitations': 'Palpitacao',
    'Dizziness': 'Tontura',
    'Swelling': 'Inchaço ',
    'Pain_Arms_Jaw_Back': 'Dor_Braco_Mandibula_Costas',
    'Cold_Sweats_Nausea': 'Suor_Frio_Nausea',
    'High_BP': 'Pressao_Alta',
    'High_Cholesterol': 'Colesterol_Alto',
    'Diabetes': 'Diabetes',
    'Smoking': 'Fumante',
    'Obesity': 'Obesidade',
    'Sedentary_Lifestyle': 'Sedentario',
    'Family_History': 'Histórico_Familiar',
    'Chronic_Stress': 'Estresse_Cronico',
    'Gender': 'Sexo',
    'Age': 'Idade',
    'Heart_Risk': 'Risco_Doenca_Cardiaca'
}
dados_cardiaco = dados_cardiaco.rename(columns=nova_coluna_nomes) # Renomeia as colunas do DataFrame

# Verificando se ha dados nulos
dados_nulos = dados_cardiaco.isnull() # Cria um DataFrame booleano indicando valores nulos
linhas_remover = dados_nulos.query('Risco_Doenca_Cardiaca == True').index # Identifica linhas com valores nulos na coluna 'Risco_Doenca_Cardiaca'
dados_cardiaco.drop(linhas_remover, axis=0, inplace=True) # Remove as linhas com valores nulos

# Treinamento
x = dados_cardiaco.drop(['Risco_Doenca_Cardiaca'], axis=1) # Separa as features (variáveis independentes)
y = dados_cardiaco.Risco_Doenca_Cardiaca # Separa a variável alvo (variável dependente)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.35) # Divide os dados em conjuntos de treino e teste
lr = LogisticRegression(max_iter=10000) # Cria o modelo de Regressão Logística com um número máximo de iterações
lr.fit(x_treino, y_treino) # Treina o modelo com os dados de treino
previsao_lr = lr.predict(x_teste) # Faz previsões com os dados de teste

acerto = accuracy_score(y_teste, previsao_lr) # Calcula a acurácia do modelo
print(f"A acertividade do modelo é: {acerto * 100:.2f}%") # Imprime a acurácia do modelo

# CRIANDO ARQUIVO PICKLE
joblib.dump(lr, 'modelo_cardiaco.pkl') # Salva o modelo treinado em um arquivo pickle
print("Modelo salvo com sucesso!") # Imprime uma mensagem de confirmação