# ABIA · Cerca local per assignar clients a centrals elèctriques

El projecte aplica Hill Climbing i Simulated Annealing d’AIMA al problema d’assignació de clients a centrals. El benefici econòmic és el criteri de la cerca principal. Els experiments de penalització permeten estats intermedis amb clients garantits pendents i validen el resultat final per separat.

## Instal·lació i comprovacions

Des de l’arrel del repositori, amb Python 3.11 i l’entorn virtual que preferiu:

```bash
python -m pip install -r requirements.txt
cd practica_busqueda_local_code
python -m unittest -v test_search_improvements
python check_state.py
python check_problem.py
```

Per regenerar l’anàlisi històrica, instal·leu també `pandas`, `scipy` i `matplotlib` i executeu `python analysis.py` des de la carpeta de la pràctica.

## Experiments

```bash
python experiments.py 7 9 --workers 1
python experiments.py 8 --workers 1
python experiments.py 4 --customers 1500 --workers 1 --output-dir results/checks
```

| ID | Comparació |
| --- | --- |
| 1–6 | Escenaris de l’enunciat amb les correccions comunes. Els IDs 30, 50 i 51 són rondes addicionals. |
| 7 | Hill Climbing amb `all` i amb `all_open_batch`. |
| 8 | Garantits amb penalització. Compara la recol·locació i la calibració de SA per separat. Les variants de SA tenen el mateix límit de 20.000 iteracions. |
| 9 | HC econòmic, refinament que redueix pèrdues a benefici constant i una variant que reprèn HC després del refinament. |

Els CSV nous es desen a `results/revised`. Els fitxers històrics es conserven a `results`. Per a una prova parcial, afegiu `--seed 1`, `--customers 1500` o `--algorithm sa` i un `--output-dir` separat. El fitxer de sortida d’un experiment es regenera en cada execució al directori triat. L’experiment 8 pot trigar força. Per provar-ne el flux amb SA i una sola llavor, es pot fer servir `python experiments.py 8 --seed 1 --algorithm sa --sa-iterations 400 --workers 1 --output-dir results/checks`.

Cada fila registra el temps d’inicialització i cerca, el temps de calibració si escau, la validesa, el benefici real i els MW de transport perduts. A SA també es registren les crides al generador, el nombre de cerques de reserva i el motiu de parada. Per comparar temps, manteniu el mateix nombre de processos i el mateix equip. Per comparar qualitat, aparelleu les execucions per llavor.

## Decisions de model i cerca

- **Parada de SA.** Després d’un mostreig fallit, es comprova si hi ha alguna acció vàlida. En cas afirmatiu es passa un únic successor a AIMA.
- **`SWAP`.** HC i SA comparteixen les mateixes parelles candidates. N’hi ha prou que un client sigui entre els deu veïns de l’altre. SA mostreja una parella i HC en comprova totes les aplicables.
- **`CLOSE`.** Es reubiquen primer els clients que han de conservar el servei. Dins de cada grup es manté l’ordre de demanda decreixent. La construcció del pla és voraç.
- **`OPEN_BATCH`.** Una central parada es pot obrir amb diversos clients en un moviment. El pla voraç prioritza el guany per MW enviat i es genera quan el grup cobreix el sobrecost d’arrencada.
- **`INSERT_RELOCATE`.** En el mode relaxat, es pot inserir un garantit bloquejat recol·locant un client del destí en una altra central activa. Es consideren fins a cinc destinacions per al garantit i es genera el primer pla factible.
- **Temperatura.** La variant calibrada estima la magnitud típica d’un empitjorament en un passeig pilot i n’ajusta la probabilitat d’acceptació inicial. Si el passeig no observa cap empitjorament, fa servir la configuració fixa.
- **`all_nearest`.** En la variant històrica el criteri és enviar la mínima potència. Entre centrals amb la mateixa pèrdua es manté el desempat per índex.
- **Pèrdues.** La fase de refinament només aplica moviments que mantenen els clients servits i les centrals actives. Redueix els MW enviats mentre conserva el benefici. La variant `resume` torna a executar la cerca econòmica des de l’assignació refinada.

La cerca principal conserva la inicialització `greedy` i l’objectiu econòmic. Els experiments 7–9 permeten mesurar els efectes de les variants noves per separat.
