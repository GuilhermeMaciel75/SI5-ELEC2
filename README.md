# **SI5-ELEC2**  

Este repositório contém o projeto desenvolvido para a disciplina **IF1014 - Mineração de Dados**, seguindo a metodologia **CRISP-DM** e utilizando o **dataset ELEC2**.  

O **ELEC2** é um conjunto de dados amplamente estudado na literatura acadêmica e contém informações sobre o mercado de eletricidade da Austrália, com medições feitas a cada 30 minutos entre os anos de **1996 e 1998**. O objetivo do projeto é analisar os dados e desenvolver modelos preditivos para estimar a variação dos preços da eletricidade, possibilitando uma melhor compreensão dos fatores que influenciam essa dinâmica.  

## **Metodologia CRISP-DM**  

A metodologia **CRISP-DM** (*Cross Industry Standard Process for Data Mining*) é um modelo amplamente utilizado para orientar projetos de **mineração de dados e ciência de dados**. Ela é composta por seis etapas, porém nesse experimento unimos algumas e tivemos um total de 4 fases:  

1. **[Compreensão do Negócio](#1-entendimento-do-negócio)** – Identificação do problema e definição dos objetivos do projeto.  
2. **[Compreensão dos Dados](#2-análise-exploratória-dos-dados-eda)** – Análise exploratória para entender a estrutura e as características do dataset.  
3. **Preparação dos Dados** – Limpeza, transformação e seleção dos dados para modelagem.  
4. **[Modelagem e Avaliação](#4-variação-paramétrica)** – Aplicação de algoritmos de aprendizado de máquina para prever a variação dos preços, e validação dos modelos preditivos para garantir sua eficácia.  

---

## **Fases do Projeto**  

### **[1. Entendimento do Negócio](./docs/1%20-%20Entendimento%20do%20Negócio.md)**  

O objetivo desta etapa foi contextualizar o **ELEC2**, explorando sua relevância no setor energético. O relatório aborda como a previsão da demanda e dos preços da eletricidade impacta empresas, investidores e órgãos reguladores.  

### **[2. Análise Exploratória dos Dados (EDA)](./docs/2%20-%20Compreenção%20dos%20Dados.md)**

**[Link do Notebook](./notebooks/02%20-%20Compreenção%20dos%20Dados.ipynb)**

Nesta fase, foram investigadas as principais características do dataset, incluindo a distribuição das variáveis e a presença de outliers. O relatório detalha a relação entre demanda, preço e transferência de energia entre regiões, além de apresentar técnicas de **redução de dimensionalidade** para identificar padrões nos dados.  

### **[4. Variação Paramétrica](./docs/4%20-%20Variação%20Paramétrica.md)**

Nesta fase do projeto, denominada **"Variação Paramétrica"**, focamos na avaliação de diferentes modelos de aprendizado de máquina e seus parâmetros, aplicados ao dataset ELEC2. O objetivo é otimizar os modelos ajustando seus hiperparâmetros para melhorar o desempenho em tarefas de classificação.

Para este experimento dividimos em Notebooks para cada um dos modelos estudados, os notebooks correspondentes são os seguintes:

- 


---

## **Contribuidores**  
- [Erlan Lira Soares Junior](https://github.com/erlanliraa)  - elsj
- [Felipe de Barros Moraes](https://github.com/FelipeMoraes03) - fbm3
- [Guilherme Maciel de Melo](https://github.com/GuilhermeMaciel75) - gmm7
- [Rubens Nascimento de Lima](https://github.com/rubdelima) - rnl2