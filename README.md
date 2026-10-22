# FlightDelayML

Rede neural escrita só com NumPy, sem framework de machine learning, para prever se um voo vai chegar atrasado. Trabalho da disciplina de Sistemas Inteligentes (Universidade Positivo), feito por Caio Willian de Melo Runpfe e Felipe Gabriel Ferreira.

## Dados

O conjunto vem do Kaggle: [February flight delay prediction](https://www.kaggle.com/datasets/divyansh22/february-flight-delay-prediction). A pasta `FlightDelayML/` tem o arquivo completo (`voos.csv`) e um recorte de cerca de 250 linhas (`tiny.csv`), que é o que o `main.py` lê.

Entradas (7):

- dia do mês (`DAY_OF_MONTH`)
- dia da semana (`DAY_OF_WEEK`)
- companhia aérea (`OP_CARRIER_AIRLINE_ID`)
- horário de partida (`DEP_TIME`)
- horário de chegada (`ARR_TIME`)
- atraso na partida (`DEP_DEL15`)
- distância (`DISTANCE`)

Saída: `ARR_DEL15`, que vale 1 quando o voo chega com 15 minutos ou mais de atraso e 0 quando não.

## Rede

Perceptron de múltiplas camadas, com 3 camadas ocultas de 40, 16 e 10 neurônios e 1 neurônio de saída. A ativação é tanh em todas as camadas e há um bias em cada uma. O treino usa retropropagação, atualizando os pesos a cada amostra. No código, a taxa de aprendizado é 0,01 e o número de épocas é 100.

## Como rodar

```bash
pip install numpy pandas matplotlib
cd FlightDelayML
python main.py
```

O script treina a rede, mostra o gráfico do erro médio por época e depois imprime as saídas e os erros de cada amostra. O teste usa as mesmas amostras do treino, então o resultado mede o ajuste aos dados e não a capacidade de generalizar.

## Relatório

O arquivo `Relatório Rede Neural.pdf` descreve os dados, o código e três testes de treino: 1000 épocas com taxa 0,01, 1000 épocas com taxa 0,07 e 2000 épocas com taxa 0,09.
