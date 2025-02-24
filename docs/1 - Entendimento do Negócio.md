# Entendimento do Negócio

## 1. Determinando o objeto do Negócio

Neste tópico, nosso objetivo é compreender o contexto de negócio no qual o dataset está inserido, analisando seu impacto na empresa e identificando oportunidades de melhoria por meio da aplicação do modelo CRISP-DM nesta base de dados. Para isso, exploraremos como os dados podem ser utilizados para otimizar processos, impulsionar a tomada de decisões estratégicas e gerar vantagens competitivas.

### 1.1. Background

A previsão da demanda de eletricidade e sua relação com a variação de preços são essenciais para decisões empresariais e governamentais. O setor energético é um dos pilares da economia moderna, impactando indústrias, serviços públicos e consumidores. Erros na previsão podem levar a falhas na oferta, penalizações regulatórias e **prejuízos financeiros.**

Nesse cenário, empresas de distribuição de energia utilizam modelos preditivos para equilibrar oferta e demanda em tempo real, reduzindo desperdícios e impactos ambientais. Logo, a volatilidade dos preços também afeta o planejamento financeiro das concessionárias e influencia investidores do setor. Dessa forma, como os preços da eletricidade são definidos em intervalos curtos (trinta minutos em nosso dataset), traders e investidores dependem de algoritmos que detectam padrões rapidamente para maximizar lucros e mitigar riscos.

Nesse contexto, a demanda por eletricidade é fortemente influenciada por fatores climáticos, como ondas de calor e frio intenso, exigindo ajustes rápidos na geração e distribuição. Assim, uma previsão precisa permite que concessionárias otimizem a infraestrutura, evitando crises energéticas. Além disso, órgãos reguladores utilizam projeções para formular políticas públicas, definir tarifas dinâmicas e incentivar o consumo eficiente de energia.

Dessa forma, no mercado de eletricidade, a precificação é altamente dinâmica, diferentemente de setores com preços mais estáveis. Alterações frequentes nos preços são impulsionadas por variáveis como oferta e demanda, sazonalidade, condições climáticas e capacidade de produção e distribuição.

Portanto, a previsão precisa do consumo e da precificação é um diferencial competitivo para empresas e um fator crítico para a estabilidade do setor. A aplicação de técnicas avançadas de aprendizado de máquina, aliadas à detecção de mudanças de conceito (**concept drift**), pode revolucionar a gestão da energia elétrica, promovendo maior segurança, eficiência e previsibilidade tanto no consumo quanto no preço desse ativo.

### 1.2. Objetivo do Negócio

Diante desse cenário volátil, a previsão precisa da demanda de eletricidade é fundamental para os diferentes agentes do mercado, incluindo operadores de redes elétricas, empresas fornecedoras de energia, gestores de usinas e investidores. Para as empresas de geração e distribuição, prever os preços permite otimizar a alocação de recursos, garantir maior eficiência na compra e venda de energia no mercado livre e reduzir riscos financeiros associados às oscilações inesperadas. Dessa forma, o uso de modelos preditivos robustos pode transformar dados históricos e variáveis externas em insights estratégicos, promovendo uma otimização na geração e distribuição de energia, reduzir desperdícios, evitar sobrecargas no sistema e mitigar riscos financeiros, podendo levar a um ganho de capital por parte da companhia.

Dessa forma, nosso objetivo é analisar o dataset ELEC2 para compreender a distribuição dos dados e identificar padrões que influenciam a precificação da energia. Além disso, buscamos desenvolver modelos preditivos capazes de determinar se o preço da eletricidade terá uma tendência de alta ou baixa, permitindo uma tomada de decisão mais estratégica e eficiente no setor.

### 1.3. Critérios de Sucesso do Objetivo de Negócio

O sucesso do projeto será medido pela melhoria do F1-score, ROC-AUC e outras métricas de avaliação nas previsões da demanda e variação dos preços da eletricidade. A escolha dessas métricas se deve ao fato de que a acurácia pode ser enganosa em cenários com dados desbalanceados e mudanças ao longo do tempo. O F1-score equilibra precisão e revocação, garantindo que os modelos sejam eficazes mesmo em classes minoritárias, enquanto o ROC-AUC avalia a capacidade do modelo de distinguir entre diferentes classes, tornando a análise mais robusta. O objetivo é gerar modelos mais confiáveis e eficazes, proporcionando suporte estratégico para os stakeholders.

## 2. Avaliando a Situação

### 2.1. Inventário de Recursos

Temos como base de dados para esse problema o ELEC2, um dataset amplamente estudado na literatura acadêmica, fornecendo informações detalhadas sobre a dinâmica do mercado de eletricidade da Austrália, entre os anos de 1996 e 1998\. Os dados abrangem um período de aproximadamente dois anos, com medições a cada 30 minutos, totalizando 45.312 registros. Cada amostra inclui 9 variáveis essenciais para o entendimento da precificação e demanda de energia, como: A **data da coleta da informação, dia da semana, a demanda de eletricidade em Nova Gales do Sul (NSW) e em Victoria, o consumo de eletricidade em NSW e Victoria e as transferências programadas entre os estados**. O principal objetivo do dataset é prever a direção da variação dos preços (UP ou DOWN), utilizando uma média móvel de 24 horas para eliminar tendências de longo prazo.

### 2.2. Requerimentos, Suposições e Restrições

A previsão de preços será baseada em dados históricos, assumindo que padrões passados podem indicar tendências futuras. No entanto, desafios como mudança de conceito (concept drift), qualidade dos dados e necessidade de alto poder computacional para o treinamento dos modelos devem ser considerados.

Já foi verificado que os dados foram previamente normalizados, facilitando as próximas etapas do CRISP-DM. O principal desafio neste contexto é o **concept drift**, que ocorre quando a relação entre as variáveis de entrada e a variável-alvo muda ao longo do tempo, reduzindo a precisão do modelo. Isso pode ocorrer devido a:

* **Mudanças sazonais e climáticas**, que impactam a demanda e o preço da eletricidade;  
* **Fatores econômicos e regulatórios**, como novas políticas de preço e mudanças na oferta de energia;  
* **Expansão do conjunto de dados**, com a inclusão de novas regiões na rede elétrica, alterando a distribuição dos dados.

Outro ponto crítico é o desbalanceamento das classes no conjunto de dados ELEC2. Ele é composto por duas classes: **UP** (aumento de preço em relação à média móvel das últimas 24 horas) e **DOWN** (diminuição). Assim, esse desbalanceamento pode comprometer a qualidade do modelo e requer técnicas específicas para mitigação.

Por fim, a definição da **janela de tempo** é essencial para o desempenho do modelo. O dataset registra preços a cada **30 minutos**, e a escolha de uma janela adequada impacta a estabilidade do aprendizado.

* **Janelas grandes** garantem mais dados para análise, mas tornam o modelo mais suscetível ao concept drift.  
* **Janelas pequenas** capturam mudanças rápidas, mas podem perder padrões relevantes em períodos de estabilidade.

Além disso, o período de previsão – próxima hora, próximo dia, etc. – influencia a forma como os padrões são identificados e o desempenho do modelo.

### 2.3. Riscos e Contingências

Como já abordado no tópico anterior, os principais riscos incluem:

* **Concept drift:** Mudanças nas relações dos dados ao longo do tempo. O que dificultaria a predição, uma vez que essas mudanças podem atrapalhar as previsões dos modelos  
* **Fatores externos:** Políticas de preço, regulações governamentais e variações econômicas.  
* **Distribuição desigual das classes:** O dataset apresenta um desequilíbrio entre classes UP e DOWN.

Medidas de contingência incluem monitoramento contínuo da performance dos modelos e adaptação de abordagens conforme necessário.

### 2.4. Terminologia

* **Concept Drift:** Mudança nos padrões dos dados ao longo do tempo.  
* **Previsão de Séries Temporais:** Modelagem baseada em dados históricos para prever tendências futuras.

### 2.5. Custos e Benefícios

O principal custo é o processamento computacional para treinamento de modelos, visto que esse dataset já foi gerado anteriormente. O seu benefício pode ser enorme, principalmente para a empresa, uma vez que ela terá uma maior previsibilidade do preço da energia e porque não da demanda, permitindo a adoção de medidas estratégicas.

## 3. Determinando os objetivos de Projeto e Data Science

Durante esse tópico, abordaremos de maneira mais técnica o planejamento e os objetivos de Data Science do projeto, a fim de delimitar objetivos e visualizar os próximos passos do projeto

### 3.1. Objetivos de Data Science

Aplicar à risca a metodologia de CRISP-DM na base ELEC2, garantindo uma abordagem estruturada para previsão de preços, utilizando e replicando algumas das técnicas propostas nos artigos estudados. Queremos realizar uma análise preliminar de dados a fim de verificar a qualidade dos dados, sua distribuição e gerar possíveis insights a partir deles. Além disso, realizar a visualização dos dados e dos resultados desses modelos, por meio de gráficos como: Gráficos de caixa (boxplot), barras e histogramas.

Além disso iremos aplicar modelos de aprendizado de máquina como **K-Nearest Neighbors (K-NN)**, **Learning Vector Quantization (LVQ), Árvore de Decisão, Support Vector Machine (SVM), Random Forest e Rede Neural MLP (Multilayer Perceptron)**. E conseguir resultados satisfatórios nas métricas de avaliação desses modelos.

### 3.2. Critério de sucesso do Data Science

O sucesso será medido através da performance dos modelos, utilizando métricas como acurácia, recall, F-1 score, matriz de confusão. Além de gerar gráficos que facilitem a tomada de decisão do stakeholder e demonstrem os desempenhos dos modelos.

## 4. Plano de Projeto

O projeto seguirá as etapas do CRISP-DM:

1. **Compreensão do Negócio:** Definir objetivos e problemas.  
2. **Compreensão dos Dados:** Analisar qualidade e distribuição.  
3. **Preparação dos Dados:** Limpeza e transformação.  
4. **Modelagem:** Aplicar técnicas de machine learning.  
5. **Avaliação:** Medir desempenho dos modelos.  
6. **Implantação:** Implementar solução para decisão empresarial.

Os prazos para o desenvolvimento de cada uma dessas etapas será atribuído pelo stakeholder.

### 4.1. Avaliação Inicial, Ferramentas e Técnicas

Para o desenvolvimento dessas etapas, será utilizado o Python como linguagem de programação e para a análise de dados e desenvolvimento de modelos de predição. O uso do Google Colab para experimentação e ambiente de execução para os scripts e o GitHub para controle de versão e armazenamento do código.

## 5. Referência teórica

### 5.1. Artigo 1 \- Performance Evaluation using Online Machine Learning Packages for Streaming Data

O artigo avalia pacotes de aprendizado de máquina online, especificamente Creme e River, aplicados a dados de streaming. Ele examina três classificadores \- Regressão Logística (LR), K-Nearest Neighbors (KNN) e Naive Bayes Gaussiano (GNB).

As bibliotecas Creme e River são apresentadas como soluções eficientes para aprendizado online. River, uma fusão entre Creme e Scikit-Multiflow, demonstrou melhor desempenho. O estudo destaca desafios como a deriva de dados e o desbalanceamento de classes, propondo soluções como janelas adaptativas e ajustes no treinamento.

Para lidar com a deriva de dados, o artigo sugere o uso de janelas adaptativas, que ajustam automaticamente a quantidade de dados recentes considerados na aprendizagem do modelo, permitindo melhor adaptação a mudanças nos padrões. Já o desbalanceamento de classes é tratado por meio da ponderação dos dados ou técnicas de amostragem, como oversampling e undersampling, garantindo que as classes minoritárias tenham impacto adequado no treinamento.

A janela de conceito é abordada como técnica essencial para a adaptação contínua dos modelos. No ELEC2, os padrões dos dados mudam ao longo do tempo, exigindo um processo de atualização dinâmico. Os resultados mostram que o KNN teve desempenho consistente, enquanto a Regressão Logística foi eficaz para lidar com variações nos dados. O estudo enfatiza que ROC-AUC e F1-score são métricas mais confiáveis que a acurácia para avaliação de modelos.

### 5.2. Artigo 2 \- Learning with Drift Detection

O artigo aborda o problema da aprendizagem de máquinas em ambientes onde a distribuição dos dados muda ao longo do tempo, um fenômeno conhecido como mudança de conceito (*concept drift)*. Normalmente, os algoritmos de aprendizado assumem que os dados são gerados de forma aleatória a partir de uma distribuição estacionária, o que nem sempre reflete a realidade de aplicações dinâmicas do mundo real.

Tendo em vista esse problema, o artigo propõe um método para detecção de mudanças na distribuição dos dados baseado no controle da taxa de erro do modelo ao longo do tempo. A ideia do método é que enquanto a distribuição dos dados for estacionária, o erro tende a diminuir com o aumento dos exemplos. No entanto, quando ocorre uma mudança na distribuição, o erro aumenta. A partir disso, o método define um nível de alerta (quando a taxa de erro ultrapassa um certo limite, o modelo fica com mais atenção para uma possível mudança) e um nível de mudança (se o erro continuar aumentando e atingir o limite de mudança, um novo contexto é declarado e o modelo precisa ser retreinado com os dados mais recentes).

Além disso, o artigo também cita outras abordagens mais comuns para lidar com essa problemática, como a utilização de janelas de tempo fixas ou utilização de pesos decrescentes para exemplos mais antigos. Também são citados métodos que monitoram indicadores de desempenho (*como accuracy, precision, recall e propriedades estatísticas dos dados*) para detectar mudanças de conceito.

A fim de avaliar o método proposto, foram realizados experimentos utilizando 3 algoritmos clássicos (Decision Tree, Redes Neurais e Perceptron) sobre conjuntos de dados sintéticos e sobre um conjunto de dados reais do mercado de eletricidade australiano (ELEC2), e os resultados obtidos demonstram que o método é eficaz na detecção de mudanças de conceito no mundo real.

Por fim, os autores concluem que o método proposto é simples e eficiente para lidar com o problema de mudança de conceito, e que esse método não é dependente dos algoritmos utilizados nos experimentos, podendo então ser generalizados para qualquer algoritmo de machine learning em diversos problemas do mundo real.

### 5.3. Artigo 3 \- Early Concept Drift Detection via Prediction Uncertainty

O artigo investiga a detecção precoce de drift de conceito em fluxos de dados, comparando diferentes métodos, com ênfase no PUDD (PU-index-based Drift Detector). O estudo avalia a eficácia do PUDD em relação a abordagens tradicionais, como ADWIN, DDM, EDDM, HDDM e KSWIN, utilizando uma gama de datasets, incluindo o ELEC2. Além disso, os experimentos foram conduzidos sob dois regimes distintos: treinamento incremental, onde o modelo é atualizado continuamente, e treinamento apenas na inicialização ou após um alerta de drift, permitindo uma análise detalhada da adaptação do modelo a mudanças conceituais.

No contexto do dataset, o PUDD demonstrou melhoria significativa em cenários com Naïve Bayes e Redes Neurais, conseguindo detectar mudanças na distribuição dos dados antes que a taxa de erro dos classificadores fosse impactada. Esse desempenho superior foi atribuído à capacidade do PUDD de monitorar a incerteza das previsões (PU-index) e aplicar um Teste Qui-Quadrado de Pearson para confirmar a relevância estatística do drift, reduzindo falsos positivos. No entanto, o método não obteve os mesmos ganhos ao ser aplicado a Decision Tree, onde o impacto do PU-index foi menos expressivo. Em geral, a melhor acurácia obtida foi de 74.93.

Dessa forma, o PUDD pode ser uma alternativa eficiente quando aplicado a modelos que se beneficiam do monitoramento da incerteza das previsões, tornando-se uma abordagem promissora para a detecção precoce de drift no dataset. Os experimentos demonstram que a dinâmica dos drifts naturais presentes no ELEC2 pode ser mais bem capturada por métodos baseados em incerteza, como o PU-index, especialmente em classificadores probabilísticos.

### 5.4 Conclusão

Os estudos analisados destacam a importância da **detecção e adaptação a mudanças de conceito (concept drift)** em fluxos de dados, especialmente na previsão de demanda e preços de eletricidade.

O primeiro artigo avalia aprendizado online com pacotes como River e Creme, mostrando como janelas adaptativas e técnicas de balanceamento de classes melhoram a precisão dos modelos. O segundo propõe um método baseado na taxa de erro para detectar concept drift e retreinar modelos automaticamente. Já o terceiro investiga a detecção precoce por meio da incerteza das previsões (PU-index), destacando o método PUDD como promissor para classificadores probabilísticos.

A integração dessas abordagens pode tornar os modelos preditivos mais robustos, melhorando a estabilidade das previsões e a eficiência do setor elétrico.

## 6. Referência Bibliográfica

J. Gama, P. Medas, G. Castillo, and P. Rodrigues. Learning with drift detection. In SBIA Brazilian Symposium on Artificial Intelligence, pages 286–295, 2004\.

G. Ditzler, R. Polikar and N. Chawla, "An Incremental Learning Algorithm for Non-stationary Environments and Class Imbalance," 2010 20th International Conference on Pattern Recognition, Istanbul, Turkey, 2010, pp. 2997-3000, doi: 10.1109/ICPR.2010.734

Santosh Kumar Ray, Seba Susan, "Performance Analysis of Online Machine Learning Frameworks for Anomaly Detection in IoT Data Streams", 2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT), pp.1-5, 2024\.

Lu, Pengqian, et al. "Early Concept Drift Detection via Prediction Uncertainty." arXiv preprint arXiv:2412.11158 (2024).
