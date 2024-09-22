# Melanoma Classification IA

### Projeto de Inteligência Artificial para Classificação de Melanoma a partir de Imagens de Pele

![Soil Drought Detection](/src/img/logo.png)

## Descrição
Este projeto utiliza técnicas de aprendizado profundo para classificar tipos de melanoma (câncer de pele) em imagens de lesões cutâneas. O objetivo é auxiliar no diagnóstico precoce e na diferenciação entre lesões benignas e malignas, contribuindo para o tratamento e prevenção do câncer de pele.

## Objetivo
Desenvolver um modelo de Inteligência Artificial capaz de identificar diferentes tipos de melanoma em imagens de pele, utilizando uma base de dados do Kaggle. O sistema ajudará profissionais de saúde a tomar decisões mais informadas e precisas.
. 

## Equipe de Desenvolvimento
- **Eric Klaus Brenner Melo e Santos**
- **Matheus Rogério Pesarini**
- **Ruan Rubino de Carvalho**

## Tecnologias Utilizadas
- **Linguagem:** Python
- **Frameworks e Bibliotecas:** 
  - TensorFlow
  - OpenCV
  - Sci-kit-learn
  - Pandas
  - NumPy
- **Dataset:** [Kaggle Soil Dataset](https://www.kaggle.com/datasets/andrewmvd/isic-2019)

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
git clone https://github.com/ericklaus16/melanoma_classification_ai
cd melanoma_classification_ia
pip install -r requirements.txt
