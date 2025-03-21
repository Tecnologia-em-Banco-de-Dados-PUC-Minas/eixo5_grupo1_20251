# *Etapa 2: Governança de Dados e Planejamento Inicial do Modelo de Dados* 
## *Planejamento de Governança de Dados* 


A governança de dados é essencial para garantir que os dados utilizados no projeto sejam acessíveis, consistentes e de alta qualidade. Em um projeto de aprendizado de máquina, como o nosso, a governança assegura que os dados sejam tratados de forma ética, eficiente e segura, além de promover a integridade dos dados ao longo de todo o ciclo de vida do projeto.

A governança de dados abrange desde a coleta, passando pela análise e processamento, até o armazenamento e compartilhamento dos dados. Para o desenvolvimento deste projeto, será implementada uma governança de dados robusta, que englobará os seguintes pontos:

### Definição de Requisitos de Dados:

*Fontes de Dados:* Os dados utilizados neste projeto vêm do conjunto de dados de imagens de ressonância magnética de tumores cerebrais, disponível no Kaggle (Fonte: [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)];

*Formato dos Dados:* O conjunto de dados contém imagens em formato .jpg, armazenadas em pastas categorizadas de acordo com os tipos de tumor cerebral (glioma, meningioma e tumor pituitário);

*Qualidade dos Dados:* A qualidade das imagens deve ser verificada, garantindo que as imagens não possuam falhas ou problemas que possam comprometer o treinamento do modelo. Também será realizada uma verificação para garantir que as imagens estão corretamente rotuladas de acordo com os tipos de tumor.

*Estrutura de Gerenciamento de Dados:* A estrutura de gerenciamento de dados será centralizada no ambiente do Kaggle, onde as imagens de ressonância magnética serão organizadas e acessadas a partir de um notebook, com o nome de projeto_db_eixo5. A governança de dados será baseada nas melhores práticas de processamento e transformação, garantindo que cada etapa de manipulação dos dados seja registrada e revisada.

*Armazenamento de Dados:* O armazenamento será realizado no ambiente do Kaggle, onde o conjunto de dados será mantido dentro de um repositório específico para este projeto.

*Acesso aos Dados:* O acesso será restrito e controlado, garantindo que apenas membros autorizados do projeto possam interagir com os dados.

### Procedimentos e Processos de Governança:

Acuracidade e Consistência dos Dados Serão realizados procedimentos contínuos de validação e verificação dos dados, a fim de garantir que as imagens estão corretamente rotuladas e que não haja dados faltantes. No processamento dos dados as imagens de ressonância magnética serão pré-processadas e normalizadas, utilizando técnicas de dimensionamento e transformação adequadas para o treinamento do modelo de machine learning. O código para processamento e análise será documentado no notebook, facilitando o rastreamento e a auditoria. Para a proteção e Privacidade dos Dados, embora os dados do Kaggle sejam públicos, será adotado um controle rigoroso para garantir que não haja vazamento de informações sensíveis sobre os pacientes, caso haja alguma forma de identificação envolvida.

### Modelo de Dados Inicial

O conjunto de dados é composto principalmente por imagens (em formato .jpg), acompanhadas de suas respectivas etiquetas (labels) indicando o tipo de tumor presente em cada imagem. E para armazena-las a estrutura inicial do banco de dados será com um modelo de dados simples e estará baseada nas imagens e seus metadados. Cada imagem será armazenada com um identificador único, e as etiquetas associadas as imagens serão categorizadas em três tipos principais: Glioma, Meningioma, e tumor pituitário. O relacionamentro entre os dados será construído com base nos seus respectivos labels para garantir que o modelo seja treinado com a associação correta entre a imagem e o tipo de tumor.



