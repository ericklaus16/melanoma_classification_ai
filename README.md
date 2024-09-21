# Soil Drought Detection IA

### Projeto de Inteligência Artificial para Detecção de Nível de Seca no Solo

![Soil Drought Detection](link_para_uma_imagem_representativa_do_projeto.png)

## Descrição
Este projeto visa criar um modelo de aprendizado profundo capaz de detectar e classificar o nível de seca em imagens de solo, utilizando redes neurais convolucionais (CNNs). A aplicação prática envolve o monitoramento agrícola e ambiental, ajudando na tomada de decisões relacionadas ao manejo de recursos hídricos e ao controle da seca.

## Objetivo
Desenvolver uma inteligência artificial treinada com uma base de dados de imagens de solo obtidas do Kaggle para detectar o nível de seca com alta precisão. 

## Equipe de Desenvolvimento
- **Eric Klaus Brenner Melo e Santos**
- **Matheus Rogério Pesarini**
- **Ruan Rubino de Carvalho**

## Tecnologias Utilizadas
- **Linguagem:** Python
- **Frameworks e Bibliotecas:** 
  - TensorFlow / Keras
  - OpenCV
  - Scikit-learn
  - Pandas
  - NumPy
- **Dataset:** [Kaggle Soil Dataset](link_para_o_dataset)

## Estrutura do Projeto
- `data/` : Contém os dados brutos e pré-processados (o conteúdo da pasta foi ignorado pelo gitignore).
- `src/` : Código-fonte principal para processamento de imagens e treino de modelos.
- `results/` : Gráficos e métricas de desempenho do modelo.
- `README.md` : Descrição do projeto.

## Instalação

### Pré-requisitos
Possuir os arquivos da base do Kaggle no diretório data/ ;
Certifique-se de que você tem o Python 3.7+ instalado. Em seguida, clone o repositório e instale as dependências:

```bash
git clone https://github.com/seu_usuario/soil-drought-detection-IA.git
cd soil-drought-detection-IA
pip install -r requirements.txt
