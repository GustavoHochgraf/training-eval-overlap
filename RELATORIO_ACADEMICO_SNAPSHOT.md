# Relatorio Academico: Snapshot Carolina x PoetaV2

Data de referencia do memo: 13 de marco de 2026.

Este documento sintetiza, em formato academico e direto, o que pode ser concluido a partir do snapshot atualmente versionado neste repositorio sobre overlap entre o corpus Carolina e tarefas de avaliacao do PoetaV2. O memo usa exclusivamente os artefatos ja presentes no repositorio, sem novas execucoes ou novos experimentos.

Base factual utilizada:

- `notebooks/training_eval_overlap.ipynb`
- `results/tables/overlap_summary_snapshot.csv`
- `results/tables/top20_suspicious_matches_snapshot.csv`

## Resumo Executivo

O snapshot atual sugere sinal baixo de overlap treino-avaliacao nas condicoes efetivamente executadas, mas ainda nao sustenta uma conclusao forte sobre contaminacao do corpus Carolina como um todo. Entre 11.409 instancias de avaliacao analisadas, apenas 15 ultrapassaram o limiar de similaridade adotado, e a leitura qualitativa dos casos mais suspeitos indica que varios deles parecem vizinhos semanticos genericos ou matches com texto ruidoso, nao duplicatas evidentes. O resultado mais defensavel, portanto, e tratar este estudo como uma auditoria exploratoria promissora, mas ainda parcial.

- 993 documentos do Carolina foram indexados no snapshot executado.
- 11.409 instancias de avaliacao foram analisadas, distribuidas em 37 tasks carregadas com sucesso.
- 15 instancias ultrapassaram o limiar de similaridade 0,85, correspondendo a 0,13% do total.
- O intervalo de confianca bootstrap de 95% para a taxa global observada foi de 0,07% a 0,20%.
- 27 das 37 tasks carregadas nao tiveram nenhum caso acima do limiar.

O que o estudo sugere: baixo sinal de overlap detectado nas condicoes testadas.  
O que o estudo ainda nao prova: ausencia de contaminacao no corpus Carolina completo, nem confirmacao de que os 15 casos encontrados sejam vazamentos reais.

## Desenho do Experimento

### Objetivo

Estimar o grau de overlap entre dados de treinamento e dados de avaliacao, buscando identificar se instancias de benchmark do PoetaV2 aparecem no corpus Carolina em grau suficiente para levantar suspeita de contaminacao.

### Metodo empregado no snapshot

O notebook executado implementa busca semantica com embeddings e recuperacao aproximada via FAISS:

- cada instancia de avaliacao do PoetaV2 e transformada em uma consulta textual;
- os documentos do Carolina sao embedados e indexados em FAISS;
- cada consulta recupera os vizinhos mais proximos;
- uma instancia e marcada como suspeita quando a similaridade cosseno top-1 e maior ou igual a 0,85.

### Configuracao efetivamente executada

- 993 documentos Carolina indexados
- 11.409 instancias de avaliacao consultadas
- 37 de 43 tasks carregadas com sucesso
- `TOP_K = 5`
- limiar de overlap `0.85`
- modelo executado: `BAAI/bge-small-en-v1.5`
- cap de `500` instancias por task
- filtro minimo de `50` caracteres por documento Carolina

### Observacao de consistencia metodologica

Ha um ponto importante de rastreabilidade: o texto introdutorio do notebook menciona `bge-m3`, mas a execucao gravada usa `BAAI/bge-small-en-v1.5`. Isso nao invalida o snapshot, mas precisa ser registrado como limitacao, porque o resultado versionado reflete um encoder mais leve e potencialmente menos robusto para recuperacao semantica em portugues.

## Resultados Quantitativos

### Panorama geral

O resultado central do snapshot e o seguinte:

- 15 positivos em 11.409 instancias
- overlap global de 0,13%
- IC bootstrap 95%: [0,07%, 0,20%]
- 27 de 37 tasks com zero positivos
- 10 tasks com ao menos um positivo

Isso aponta para um sinal global baixo. Mesmo sem discutir ainda a qualidade dos matches, a contagem absoluta de casos acima do limiar e pequena demais para sustentar uma narrativa forte de contaminacao disseminada.

### Concentracao do sinal

Os cinco grupos com maior numero de casos positivos respondem por 66,67% dos 15 positivos observados:

- `bluex_launch_version_`: 3 casos
- `enem_greedy`: 2 casos
- `enem_2022_greedy`: 2 casos
- `mina_br_greedy`: 2 casos
- `bigbench_pt_simple_ethical_questions_greedy`: 1 caso

Essa concentracao importa por dois motivos:

- o sinal nao esta espalhado de forma uniforme entre as tasks;
- parte da evidencia aparente esta concentrada em familias de tasks potencialmente aparentadas, o que pede cuidado antes de tratar cada ocorrencia como evidencia independente.

### Tasks com overlap nao nulo

| Task | Positivos | Total | Taxa (%) | IC 95% |
| --- | ---: | ---: | ---: | --- |
| `enem_greedy` | 2 | 180 | 1,11 | [0,00; 2,78] |
| `enem_2022_greedy` | 2 | 180 | 1,11 | [0,00; 2,78] |
| `bigbench_pt_simple_ethical_questions_greedy` | 1 | 115 | 0,87 | [0,00; 2,61] |
| `bluex_launch_version_` | 3 | 500 | 0,60 | [0,00; 1,40] |
| `mina_br_greedy` | 2 | 500 | 0,40 | [0,00; 1,00] |
| `assin_sts_greedy` | 1 | 500 | 0,20 | [0,00; 0,60] |
| `mkqa_greedy` | 1 | 500 | 0,20 | [0,00; 0,60] |
| `pt_hate_speech_greedy` | 1 | 500 | 0,20 | [0,00; 0,60] |
| `assin_rte_greedy` | 1 | 500 | 0,20 | [0,00; 0,60] |
| `storycloze_pt_greedy` | 1 | 500 | 0,20 | [0,00; 0,60] |

### Leitura estatistica adequada

Algumas cautelas sao importantes ao interpretar a tabela:

- os intervalos de confianca por task sao amplos, especialmente quando ha poucos positivos;
- varias tasks com taxa positiva muito baixa diferem entre si por 1 ou 2 exemplos, o que nao autoriza hierarquias fortes entre benchmarks;
- a taxa geral e baixa, mas o snapshot e parcial, entao o numero nao deve ser extrapolado para o corpus Carolina completo;
- a separacao entre tasks positivas e negativas nao e nitida apenas pela similaridade media: por exemplo, `wsc285_pt_greedy` teve similaridade top-1 media de 0,80 e ainda assim zero casos acima do limiar. Isso mostra que similaridade media alta, sozinha, nao significa contaminacao.

### Figuras do snapshot

Distribuicao das similaridades e taxa de overlap por task:

![Distribuicao de similaridades e taxa de overlap](results/figures/overlap_distribution_snapshot.png)

Heatmap de overlap por task:

![Heatmap de overlap por task](results/figures/overlap_heatmap_snapshot.png)

## Leitura Qualitativa dos Matches Suspeitos

Os 20 casos de maior similaridade armazenados no snapshot ajudam a interpretar a natureza do sinal detectado. A leitura qualitativa sugere que muitos hits mais fortes nao sao duplicatas evidentes entre benchmark e treino, mas aproximacoes semanticas com textos narrativos ou ruidosos.

### Padrao geral observado

Tres padroes aparecem com clareza:

- varios matches envolvem trechos que parecem legendas, dialogos, transcricoes ou material narrativo de baixa confiabilidade documental;
- o mesmo texto recuperado reaparece em varios casos do top-20;
- algumas ocorrencias se repetem em tasks relacionadas, o que reduz a independencia aparente dos achados.

No top-20 suspeito, ha apenas 11 passagens recuperadas distintas. Em outras palavras, 20 casos suspeitos se apoiam em um conjunto relativamente pequeno de matches recorrentes. Dois blocos sao particularmente expressivos:

- um mesmo trecho de legenda associado a `Equipe AceSubs ... A MENINA SEM MAOS` aparece 4 vezes, ligado a variantes de `assin_rte_greedy` e `assin_sts_greedy`;
- um mesmo trecho narrativo iniciado por `Wow! Incrivel! O que um belo mar!` aparece 4 vezes, ligado a variantes de `enem_greedy` e `enem_2022_greedy`.

Esse comportamento enfraquece uma leitura de diversidade ampla de contaminacao e reforca a hipotese de matches repetitivos, possivelmente induzidos por material ruidoso do corpus.

### Exemplos ilustrativos

| Caso | Similaridade | Leitura qualitativa |
| --- | ---: | --- |
| `bluex_launch_version__328` | 0,8696 | Prompt literario do benchmark aproximado de prosa narrativa nao obviamente relacionada, sem indicio claro de copia literal. |
| `storycloze_pt_greedy_316` | 0,8548 | Mini-historia sobre Natal aproximada de um trecho que parece legenda ou roteiro do Futurama; semanticamente relacionado, mas nao duplicata evidente. |
| `enem_greedy_37` | 0,8574 | Trecho de leitura do ENEM aproximado de texto com estilo de legenda/dialogo audiovisual; o match parece estranho para uma interpretacao de vazamento direto. |
| `assin_rte_greedy_2284` | 0,8545 | Par de sentencas curtas aproximado de um trecho narrativo de legenda; a relacao semantica nao e clara o suficiente para chamar de contaminacao confirmada. |
| `mkqa_greedy_1879` | 0,8539 | Pergunta sobre festivais de Natal aproximada de outro trecho recorrente de narrativa/legenda, mais compativel com vizinhanca semantica generica do que com sobreposicao textual forte. |

### Conclusao qualitativa

O top-20 reforca a necessidade de validacao manual. Nas condicoes atuais, um score acima de 0,85 parece ser um bom gatilho para triagem, mas nao um criterio suficiente para classificar automaticamente uma ocorrencia como contaminacao real.

## Conclusoes que o Snapshot Permite Tirar

### Conclusoes validas

Com base apenas no snapshot versionado, as seguintes conclusoes sao defensaveis:

- o experimento nao mostra sinal forte de overlap treino-avaliacao;
- o metodo encontra poucos casos acima do limiar nas condicoes executadas;
- uma parcela relevante dos casos mais fortes parece ambigua sob inspecao qualitativa;
- o resultado global e mais consistente com "baixo sinal detectado" do que com "contaminacao disseminada";
- o pipeline ja e suficiente para funcionar como prova de conceito de auditoria semantica.

### Conclusoes que nao devem ser feitas

As seguintes afirmacoes nao sao suportadas por este snapshot:

- `nao ha contaminacao no Carolina`;
- `a avaliacao do PoetaV2 esta limpa de forma definitiva`;
- `os 15 casos encontrados sao 15 vazamentos confirmados`;
- `a taxa observada de 0,13% representa o corpus Carolina completo`.

Em outras palavras, o snapshot permite uma conclusao preliminar conservadora, mas nao uma sentenca final sobre contaminacao.

## Limitacoes e Ameacas a Validade

Esta secao precisa ser lida como parte central do resultado, nao como nota de rodape.

- Apenas 993 documentos do Carolina entraram na indexacao executada. Isso e um subconjunto pequeno demais para representar o corpus completo.
- O notebook usou `MAX_CAROLINA_DOCS = 1000`, e apenas 993 documentos passaram no filtro de tamanho minimo.
- Houve truncamento de 500 instancias por task, o que reduz cobertura e pode alterar as taxas observadas.
- Seis tasks falharam no carregamento: `agnews_pt_greedy`, `boolq_pt_greedy`, `imdb_pt_greedy`, `sst2_pt_greedy`, `hatebr_binary_greedy` e `massive_greedy`.
- O modelo efetivamente executado foi `BAAI/bge-small-en-v1.5`, mais leve do que o modelo anunciado no texto introdutorio do notebook.
- O estudo adotou um unico limiar de 0,85 sem calibracao formal baseada em adjudicacao humana.
- Nao houve rotulacao humana sistematica dos hits; houve apenas inspecao qualitativa dos casos mais suspeitos.
- Parte dos resultados suspeitos parece repetir familias de tasks aparentadas, como `enem_greedy`/`enem_2022_greedy` e `assin_rte_greedy`/`assin_sts_greedy`, reduzindo a independencia entre ocorrencias.
- O corpus recuperado parece conter muito texto ruidoso, incluindo passagens com estilo de legenda, dialogo ou transcricao. Isso pode inflar similaridade sem indicar vazamento real de benchmark.
- Como o snapshot e parcial, o estudo e mais apropriado para detectar sinais iniciais e falhas de metodo do que para estabelecer estimativas finais de contaminacao.

## Proximos Passos Recomendados

Para transformar este snapshot em uma auditoria robusta, a ordem de prioridade recomendada e:

1. Rodar o experimento no corpus Carolina completo.
2. Remover o cap de 500 instancias por task.
3. Corrigir o carregamento das 6 tasks faltantes.
4. Repetir a busca com `bge-m3` ou outro encoder mais forte para portugues ou cenario multilingue.
5. Fazer adjudicacao manual dos hits mais fortes, com criterios explicitos de verdadeiro positivo, falso positivo e caso inconclusivo.
6. Agrupar resultados por familia de benchmark para evitar interpretar variantes quase identicas como evidencias independentes.

O principal ganho desses passos e simples: sair de um snapshot exploratorio para uma auditoria metodologicamente defensavel perante banca, orientador e grupo de pesquisa.

## Sintese Final

O estado atual da evidencia aponta para baixo sinal de overlap detectado, com varios indicios de que parte dos matches mais fortes e explicavel por vizinhanca semantica generica ou por ruido do corpus, e nao por vazamento evidente de benchmark. Isso e um resultado relevante, porque reduz o risco de superinterpretacao imediata de contaminacao. Ao mesmo tempo, a cobertura parcial do Carolina, o uso de um encoder leve, o cap por task e as falhas de carregamento impedem qualquer conclusao definitiva. A leitura correta, hoje, e: o pipeline parece promissor, os primeiros resultados sao relativamente tranquilizadores, mas ainda nao ha base suficiente para uma afirmacao final sobre contaminacao no Carolina x PoetaV2.

## Apendice A: Tasks com Overlap Positivo

| Task | Positivos | Total | Taxa (%) |
| --- | ---: | ---: | ---: |
| `enem_greedy` | 2 | 180 | 1,11 |
| `enem_2022_greedy` | 2 | 180 | 1,11 |
| `bigbench_pt_simple_ethical_questions_greedy` | 1 | 115 | 0,87 |
| `bluex_launch_version_` | 3 | 500 | 0,60 |
| `mina_br_greedy` | 2 | 500 | 0,40 |
| `assin_sts_greedy` | 1 | 500 | 0,20 |
| `mkqa_greedy` | 1 | 500 | 0,20 |
| `pt_hate_speech_greedy` | 1 | 500 | 0,20 |
| `assin_rte_greedy` | 1 | 500 | 0,20 |
| `storycloze_pt_greedy` | 1 | 500 | 0,20 |

## Apendice B: Tasks que Falharam no Snapshot

- `agnews_pt_greedy`
- `boolq_pt_greedy`
- `imdb_pt_greedy`
- `sst2_pt_greedy`
- `hatebr_binary_greedy`
- `massive_greedy`
