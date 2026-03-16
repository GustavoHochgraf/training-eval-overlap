# Relatorio Academico Completo: Carolina x PoetaV2

## Resumo Executivo

Este estudo investigou a relacao entre o corpus Carolina, usado como base de treino em portugues, e o benchmark PoetaV2, usado como avaliador. O resultado principal nao foi uma prova de contaminacao nem uma prova de ausencia de overlap. O achado mais robusto foi metodologico: a conclusao sobre overlap mudou fortemente conforme o encoder, o threshold e o preprocessamento utilizados. Em outras palavras, a auditoria Carolina x PoetaV2 mostrou alta sensibilidade de desenho experimental e, por isso, exige leitura conservadora.

### Quadro Geral Das Rodadas

| Rodada | Modelo | Docs Carolina | Tasks carregadas | Instancias | Threshold | Overlaps | Taxa | Tasks com overlap > 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Snapshot exploratorio | `BAAI/bge-small-en-v1.5` | 993 | 37/43 | 11.409 | 0,85 | 15 | 0,13% | 10 |
| Rerun multilingue principal | `BAAI/bge-m3` | 993 | 37/43 | 11.409 | 0,85 | 0 | 0,00% | 0 |
| Rerun multilingue alternativo | `intfloat/multilingual-e5-large-instruct` | 993 | 37/43 | 11.409 | 0,85 | 3.527 | 30,91% | 34 |

Fonte consolidada: `results/tables/study_run_summary.csv`.

## 1. Pergunta De Pesquisa E Motivacao

O problema deste estudo e especifico do pipeline do projeto:

- o modelo em portugues e treinado sobre o corpus Carolina;
- o desempenho e acompanhado por meio do benchmark PoetaV2;
- portanto, a avaliacao depende da relacao entre o universo do corpus e o universo do benchmark.

Essa relacao pode produzir pelo menos dois riscos cientificos:

- **risco de proximidade excessiva**: se PoetaV2 estiver muito proximo do material do Carolina, o benchmark pode superestimar o quanto o modelo aprendeu capacidades gerais;
- **risco de cobertura enviesada**: se PoetaV2 capturar apenas uma parte estreita do que o modelo deveria aprender, a avaliacao pode ser metodologicamente limitada mesmo sem haver vazamento literal.

Por isso, a pergunta central aqui nao e apenas "ha contaminacao?". A pergunta mais importante e:

> o benchmark PoetaV2 esta suficientemente dissociado do universo textual do Carolina para servir como evidencia robusta de melhoria do modelo?

## 2. Desenho Experimental

### 2.1 Metodologia comum

As tres rodadas seguiram a mesma estrutura conceitual:

1. indexar documentos do Carolina;
2. extrair instancias de avaliacao do PoetaV2;
3. gerar embeddings para documentos e queries;
4. realizar busca vetorial top-k com FAISS;
5. marcar como "suspeito" todo caso acima de um threshold de similaridade;
6. inspecionar os casos mais fortes qualitativamente.

Configuracao comum das rodadas registradas:

- `993` documentos Carolina indexados;
- `11.409` instancias PoetaV2;
- `37` tasks carregadas com sucesso de um total tentado de `43`;
- busca `top-5`;
- threshold nominal de overlap `0,85`.

### 2.2 Rodada 1: snapshot exploratorio

Objetivo:

- validar o pipeline de ponta a ponta rapidamente;
- obter um primeiro sinal de overlap semantico.

Configuracao relevante:

- modelo: `BAAI/bge-small-en-v1.5`;
- `MAX_CAROLINA_DOCS = 1000`;
- `MAX_INSTANCES_PER_TASK = 500`;
- sem truncation explicita registrada para documentos.

Interpretacao correta:

- essa rodada serviu como baseline exploratorio;
- ela nao deveria ser tratada como evidencia final para portugues, porque o encoder era ingles-centrado.

### 2.3 Rodadas 2 e 3: reruns multilingues

Motivacao:

- o corpus e o benchmark sao majoritariamente em portugues;
- era necessario repetir a auditoria com modelos semanticamente mais adequados para contexto multilingue.

Modelos testados:

- `BAAI/bge-m3`
- `intfloat/multilingual-e5-large-instruct`

Configuracao local explicitamente registrada:

- `MAX_CAROLINA_DOCS = 1000`
- `MAX_INSTANCES_PER_TASK = 500`
- `model_max_seq_length = 512`
- `max_document_chars = 2000`
- `batch_size = 16` para `bge-m3`
- `batch_size = 8` para `multilingual-e5-large-instruct`

### 2.4 Por que truncation foi adicionada

Os reruns recentes nasceram apos um diagnostico operacional importante:

- os documentos Carolina eram muito longos em forma bruta;
- no subconjunto usado, o comprimento mediano era muito alto e o percentil 95 passava de dezenas de milhares de caracteres;
- `bge-m3` vinha com `max_seq_length` padrao muito alto para uma execucao local confortavel;
- o primeiro batch estava concentrando muito custo computacional.

Por isso, a rodada local passou a registrar explicitamente:

- truncation em caracteres antes da tokenizacao;
- reducao do `max_seq_length`;
- metadados salvos por run para tornar essa escolha transparente.

Essa escolha melhora reprodutibilidade e throughput, mas tambem entra como limitacao metodologica do estudo.

## 3. Resultados Consolidados

### 3.1 Visao geral

Os tres experimentos produziram regimes muito diferentes:

- Snapshot `bge-small-en-v1.5`: `15/11.409` casos acima de `0,85` (`0,13%`)
- `bge-m3`: `0/11.409` casos acima de `0,85` (`0,00%`)
- `multilingual-e5-large-instruct`: `3.527/11.409` casos acima de `0,85` (`30,91%`)

Figura sintese:

![Taxa de overlap por rodada](results/figures/overall_overlap_rate_comparison.png)

### 3.2 Tasks mais afetadas no rerun `multilingual-e5-large-instruct`

| Task | Overlaps | Total | Taxa |
| --- | ---: | ---: | ---: |
| `broverbs_mc_greedy` | 24 | 27 | 88,89% |
| `tweetsentbr_greedy` | 385 | 500 | 77,00% |
| `storycloze_pt_greedy` | 372 | 500 | 74,40% |
| `bigbench_pt_causal_judgment_greedy` | 139 | 190 | 73,16% |
| `bigbench_pt_analogical_similarity_greedy` | 217 | 300 | 72,33% |
| `mina_br_greedy` | 360 | 500 | 72,00% |
| `wsc285_pt_greedy` | 198 | 285 | 69,47% |
| `bigbench_pt_social_iqa_greedy` | 174 | 300 | 58,00% |
| `bigbench_pt_bbq_greedy` | 275 | 500 | 55,00% |
| `pt_hate_speech_greedy` | 274 | 500 | 54,80% |

Leitura:

- o sinal do `e5` nao foi localizado em poucas tasks;
- ele se espalhou por grande parte do benchmark;
- mesmo assim, isso nao basta para concluir contaminacao, porque a distribuicao de similaridade do modelo ficou muito comprimida para cima.

### 3.3 Maiores diferencas entre `bge_m3` e `e5`

| Task | `bge-m3` | `e5` | Delta |
| --- | ---: | ---: | ---: |
| `broverbs_mc_greedy` | 0,00% | 88,89% | +88,89 pp |
| `tweetsentbr_greedy` | 0,00% | 77,00% | +77,00 pp |
| `storycloze_pt_greedy` | 0,00% | 74,40% | +74,40 pp |
| `bigbench_pt_causal_judgment_greedy` | 0,00% | 73,16% | +73,16 pp |
| `bigbench_pt_analogical_similarity_greedy` | 0,00% | 72,33% | +72,33 pp |
| `mina_br_greedy` | 0,00% | 72,00% | +72,00 pp |
| `wsc285_pt_greedy` | 0,00% | 69,47% | +69,47 pp |
| `bigbench_pt_social_iqa_greedy` | 0,00% | 58,00% | +58,00 pp |
| `bigbench_pt_bbq_greedy` | 0,00% | 55,00% | +55,00 pp |
| `pt_hate_speech_greedy` | 0,00% | 54,80% | +54,80 pp |

Fonte: `results/tables/top_task_deltas_summary.csv`.

### 3.4 Sensibilidade do threshold

O comportamento por threshold deixa claro que `0,85` nao e uma escala comparavel entre os dois modelos.

| Modelo | >=0,80 | >=0,82 | >=0,85 | >=0,88 | >=0,90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3` | 0,00% | 0,00% | 0,00% | 0,00% | 0,00% |
| `multilingual-e5-large-instruct` | 97,95% | 84,84% | 30,91% | 0,60% | 0,02% |

Fonte: `results/tables/threshold_sensitivity_summary.csv`.

Isso sugere duas coisas ao mesmo tempo:

- o `bge-m3` ficou inteiramente abaixo do threshold herdado;
- o `e5` ficou com a massa de similaridade concentrada acima dele, mas despencando quando o threshold sobe um pouco.

## 4. Analise Interpretativa

### 4.1 O que mudou quando passamos para modelos multilingues

O rerun multilingue foi necessario e correto do ponto de vista linguistico. O problema e que ele revelou alta instabilidade metodologica:

- `bge-m3` teve media de similaridade `0,4929` e maximo aproximado de `0,7354`;
- `e5` teve media de similaridade `0,8394` e maximo aproximado de `0,9264`;
- mantendo o mesmo threshold (`0,85`), os modelos passaram a responder de formas qualitativamente incompatíveis.

Consequencia:

- o threshold deixou de ser um criterio comparavel entre runs;
- o encoder virou parte decisiva da propria definicao operacional de "overlap".

### 4.2 O que isso significa para Carolina x PoetaV2

O estudo passa a sustentar uma leitura mais sofisticada:

- nao basta perguntar se ha ou nao ha overlap;
- e preciso perguntar como esse overlap esta sendo medido e com que encoder ele aparece.

O resultado recente sugere que:

- ha proximidade semantica suficiente para produzir muitos pares de alta similaridade sob um encoder;
- essa proximidade pode refletir similaridade generica de genero textual, dialogo, narrativa e subtitles, e nao necessariamente compartilhamento de conteudo avaliativo relevante;
- o benchmark pode estar capturando padroes que tangenciam o universo do Carolina, mas o tamanho exato desse efeito ainda depende fortemente da calibracao do metodo.

### 4.3 Interpretacao moderada do achado

O achado forte deste estudo e:

> a inferencia "Carolina e PoetaV2 sao proximos" ou "Carolina e PoetaV2 sao distantes" ainda nao e robusta sem calibracao model-wise e revisao qualitativa sistematica.

Esse e um resultado util para o projeto porque impede tanto um otimismo falso quanto um alarme falso.

## 5. Leitura Qualitativa Dos Matches

### 5.1 Exemplos do snapshot inicial

Os matches mais fortes da rodada inicial ja sugeriam ambiguidade:

| Task | Similaridade | Leitura qualitativa |
| --- | ---: | --- |
| `bluex_launch_version_` | 0,8696 | prompt literario casando com prosa narrativa sem duplicata evidente |
| `mina_br_greedy` | 0,8585 | fala toxica casando com dialogo/subtitulo de outro contexto |
| `enem_greedy` | 0,8574 | texto de leitura casando com trecho de estilo anime/subtitulo |
| `storycloze_pt_greedy` | 0,8548 | mini-historia natalina casando com fragmento narrativo generico de Natal |

Esse padrao ja apontava necessidade de validacao manual antes de chamar qualquer caso de contaminacao real.

### 5.2 Hub documents no rerun `e5`

Um dos sinais mais importantes do rerun `e5` foi a concentracao dos hits em poucos documentos Carolina:

| `top1_doc_id` | Hits | Tasks distintas | Similaridade media | Preview |
| --- | ---: | ---: | ---: | --- |
| `68` | 192 | 21 | 0,8630 | dialogo escolar com tom de legenda/transcricao |
| `662` | 157 | 22 | 0,8605 | dialogo coloquial sobre escola/beijo/germes |
| `929` | 131 | 21 | 0,8590 | repeticao de falas no estilo subtitle |
| `553` | 97 | 11 | 0,8621 | recap narrativo e fala serializada |
| `52` | 92 | 12 | 0,8576 | dialogo cotidiano e texto publicitario |
| `814` | 86 | 6 | 0,8662 | subtitle com fala coloquial e humor |
| `614` | 85 | 17 | 0,8567 | dialogo narrativo com tom de roteiro |
| `421` | 76 | 10 | 0,8621 | recap de episodio com fala dramatica |

Fonte completa: `results/tables/e5_hub_documents.csv`.

Interpretacao:

- os hits nao parecem distribuídos uniformemente pelo Carolina;
- varios se concentram em textos com cara de legenda, sitcom, roteiro, recap ou transcricao;
- isso e compativel com efeito de **hubness semantica** e com falso-positivo por genero textual generico.

### 5.3 Consequencia para a leitura dos 3.527 casos do `e5`

Esses `3.527` casos devem ser tratados como:

- **casos suspeitos para auditoria**, e nao
- **casos confirmados de vazamento**.

Sem adjudicacao humana, a interpretacao correta e de sinal inflado e heterogeneo.

## 6. Limitacoes E Ameacas A Validade

### 6.1 Limitacoes de corpus e amostra

- apenas `993` documentos Carolina foram usados;
- o corpus completo nao foi auditado;
- cada task PoetaV2 foi truncada em `500` instancias;
- `37` de `43` tasks foram carregadas com sucesso.

Tasks que falharam no snapshot:

- `agnews_pt_greedy`
- `boolq_pt_greedy`
- `imdb_pt_greedy`
- `sst2_pt_greedy`
- `hatebr_binary_greedy`
- `massive_greedy`

### 6.2 Limitacoes de modelo

- o snapshot inicial usou um encoder exploratorio ingles-centrado;
- os dois modelos multilingues recentes apresentaram comportamentos drasticamente diferentes sob o mesmo threshold;
- nao houve calibracao supervisionada do threshold por modelo;
- portanto, comparacoes de taxa absoluta entre runs precisam ser tratadas com cautela.

### 6.3 Limitacoes de preprocessamento

- os reruns locais usaram truncation explicita: `max_seq_length = 512` e `max_document_chars = 2000`;
- isso foi uma escolha operacional necessaria, mas altera a unidade efetiva de comparacao;
- ainda nao ha chunking em passagens, que seria uma formulacao mais adequada para auditoria de overlap.

### 6.4 Limitacoes de inferencia

- similaridade alta nao prova contaminacao;
- nao houve rotulacao humana sistematica dos casos suspeitos;
- muitos hits fortes parecem vizinhanca semantica generica;
- houve forte concentracao em poucos documentos Carolina, o que enfraquece a leitura de contaminacao distribuida.

## 7. Conclusoes Validas

O que este estudo permite afirmar:

- houve alta instabilidade metodologica entre encoders;
- o benchmark PoetaV2 mostrou forte sensibilidade a encoder e threshold;
- a massa de textos Carolina parece conter bastante material narrativo generico, dialogado e potencialmente ruidoso para esse tipo de auditoria;
- a rodada `e5` indica que existe proximidade semantica detectavel sob certos encoders, mas o significado exato dessa proximidade ainda e ambiguo;
- o estudo ainda nao permite afirmar contaminacao real de PoetaV2 pelo Carolina.

## 8. Conclusoes Invalidas

O que este estudo nao permite afirmar:

- que PoetaV2 esta definitivamente limpo;
- que PoetaV2 esta definitivamente contaminado;
- que os `3.527` casos do `e5` sejam todos vazamento;
- que `0` casos no `bge-m3` provem ausencia de overlap relevante;
- que o threshold `0,85` seja um criterio universal e comparavel entre modelos.

## 9. Proximos Passos Recomendados

Prioridade metodologica:

1. calibrar threshold por modelo com amostra anotada manualmente;
2. substituir truncation simples por chunking em passagens com stride;
3. rerodar no corpus Carolina completo;
4. remover o cap de `500` instancias por task;
5. revisar as `6` tasks que falharam;
6. agrupar benchmarks por familia para evitar contar variantes proximas como evidencias independentes.

Meta do proximo experimento:

> transformar este estudo de um snapshot exploratorio sensivel ao encoder em uma auditoria robusta e comparavel entre modelos.

## 10. Artefatos Do Estudo

Artefatos principais para leitura:

- resumo executivo: `README.md`
- comparacao geral das rodadas: `results/tables/study_run_summary.csv`
- sensibilidade por threshold: `results/tables/threshold_sensitivity_summary.csv`
- hub documents do `e5`: `results/tables/e5_hub_documents.csv`
- maiores deltas por task: `results/tables/top_task_deltas_summary.csv`
- comparacao entre runs: `results/runs/comparison/model_comparison.md`

Notebook e infraestrutura:

- snapshot inicial: `notebooks/training_eval_overlap.ipynb`
- rerun multilingue: `notebooks/training_eval_overlap_multilingual_rerun.ipynb`
- gerador dos artefatos consolidados: `scripts/build_study_report_artifacts.py`

## Fechamento

Se a pergunta for "podemos usar PoetaV2 como evidencia forte de melhora do modelo treinado em Carolina?", a resposta atual e: **ainda com cautela**. O estudo mostrou que a relacao Carolina x PoetaV2 nao e trivial e que a auditoria depende fortemente de escolhas metodologicas. Esse resultado, por si so, ja e relevante para orientar a proxima fase do projeto.
