# Importar a biblioteca
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import pandas as pd 
import joblib 

print("\n")
print("   ██████████████████████████████████████  ")  # Azul
print("    🩺  ANÁLISE CARDÍACA RÁPIDA  🩺        ")  # Branco
print("   ██████████████████████████████████████  ")  # Azul
print("\n")

def validar_resposta(pergunta):

#1 se a resposta for "Sim", 0 se for "Não".
   
    while True:
        resposta = input(pergunta).strip().lower().replace('nao', 'não')
        if resposta in ["sim", "não"]:
            return 1 if resposta == "sim" else 0
        else:
            print("Resposta inválida. Por favor, responda com 'Sim' ou 'Não'.")

while True:  # Loop principal para múltiplas previsões
    # Coleta de dados do usuário
    dor_peito = validar_resposta("Você sente dor no peito? (Sim/Não): ")
    falta_ar = validar_resposta("Você tem falta de ar frequente? (Sim/Não): ")
    fadiga = validar_resposta("Você se sente cansado(a) ou fatigado(a) com frequência? (Sim/Não): ")
    palpitacao = validar_resposta("Você sente palpitações ou batimentos cardíacos irregulares? (Sim/Não): ")
    tontura = validar_resposta("Você tem tonturas frequentes? (Sim/Não): ")
    inchaco = validar_resposta("Você tem inchaço nos pés, tornozelos ou pernas? (Sim/Não): ")
    dor_braco = validar_resposta("Você sente dor no braço, mandíbula ou costas? (Sim/Não): ")
    suor_nausea = validar_resposta("Você tem suores frios ou náuseas frequentes? (Sim/Não): ")
    pressao_alta = validar_resposta("Você tem pressão alta? (Sim/Não): ")
    colesterol_alto = validar_resposta("Você tem colesterol alto? (Sim/Não): ")
    diabetes = validar_resposta("Você tem diabetes? (Sim/Não): ")
    fumante = validar_resposta("Você é fumante? (Sim/Não): ")
    obesidade = validar_resposta("Você é obeso(a)? (Sim/Não): ")
    sedentario = validar_resposta("Você é sedentário(a)? (Sim/Não): ")
    historico_familiar = validar_resposta("Você tem histórico familiar de doenças cardíacas? (Sim/Não): ")
    estresse_cronico = validar_resposta("Você tem estresse crônico? (Sim/Não): ")

    # Coleta do sexo do usuário
    while True:
        sexo_input = input("Qual o seu sexo? (Masculino/Feminino): ").strip().lower()
        if sexo_input in ["masculino", "feminino"]:
            sexo = 1 if sexo_input == "masculino" else 0
            break
        else:
            print("Resposta inválida. Por favor, responda com 'Masculino' ou 'Feminino'.")

    # Coleta da idade do usuário
    while True:
        try:
            idade = int(input("Qual a sua idade? (Em anos): "))
            break
        except ValueError:
            print("Idade inválida. Por favor, insira um número inteiro.")

    # Criação do DataFrame com os dados do usuário
    dados_usuario = {
        'Dor_no_Peito': [dor_peito],
        'Falta_de_Ar': [falta_ar],
        'Fadiga': [fadiga],
        'Palpitacao': [palpitacao],
        'Tontura': [tontura],
        'Inchaço ': [inchaco],
        'Dor_Braco_Mandibula_Costas': [dor_braco],
        'Suor_Frio_Nausea': [suor_nausea],
        'Pressao_Alta': [pressao_alta],
        'Colesterol_Alto': [colesterol_alto],
        'Diabetes': [diabetes],
        'Fumante': [fumante],
        'Obesidade': [obesidade],
        'Sedentario': [sedentario],
        'Histórico_Familiar': [historico_familiar],
        'Estresse_Cronico': [estresse_cronico],
        'Sexo': [sexo],
        'Idade': [idade]
    }

    # Carrega o modelo treinado
    modelo_treinado = joblib.load('modelo_cardiaco.pkl') # ajuste o caminho

    # Cria o DataFrame para a previsão
    dados_usuario = pd.DataFrame(dados_usuario)

    # Realiza a previsão
    previsao = modelo_treinado.predict(dados_usuario)

    # Exibe o resultado da previsão
    if previsao[0] == 1:
        print("VOCÊ TEM TENDÊNCIA A TER PROBLEMAS CARDÍACOS, PROCURE UM MÉDICO!!")
    else:
        print("De acordo com suas respostas, você não tem tendência a ter problemas cardíacos. Continue mantendo hábitos saudáveis!")

    # Pergunta se o usuário deseja realizar outra previsão
    while True:
        continuar = input("Deseja realizar outra previsão? (Sim/Não): ").strip().lower().replace('nao', 'não')
        if continuar in ["sim", "não"]:
            break
        else:
            print("Resposta inválida. Por favor, responda com 'Sim' ou 'Não'.")

    if continuar == "não":
        break  # Sai do loop se o usuário não quiser continuar

print("Programa finalizado.")


