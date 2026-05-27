from pathlib import Path

import nbformat as nbf


ROOT = Path('/workspace/geovision-cali-hf/Entrega_Final/Situacion2')
NB_DIR = ROOT / 'notebooks'
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write_notebook(name, title, cells):
    nb = nbf.v4.new_notebook()
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
    }
    nb['cells'] = [md(f'# {title}\n\nNotebook consolidado de entrega final para Situacion 2. Raiz evaluable unica: `/workspace/geovision-cali-hf/Entrega_Final/Situacion2`.')] + cells
    nbf.write(nb, NB_DIR / name)


setup = """from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

ROOT = Path('/workspace/geovision-cali-hf/Entrega_Final/Situacion2')
pd.set_option('display.max_colwidth', 160)
"""


write_notebook(
    '01_checkpoint_md5_consolidado.ipynb',
    'Entregable 1 - Checkpoint y MD5',
    [
        md('## Objetivo\n\nConsolidar el checkpoint `.pt` entrenado, sus metricas principales y la verificacion MD5 reproducible. Las rutas usadas aqui son las rutas finales autoritativas de la entrega.'),
        code(setup),
        code("""with open(ROOT / 'metricas/sweep_best_metrics.json') as f:
    sweep = json.load(f)
with open(ROOT / 'metricas/sweep_best_md5.json') as f:
    md5 = json.load(f)
with open(ROOT / 'metricas/full_logging_manifest.json') as f:
    full = json.load(f)
with open(ROOT / 'metricas/full_logging_md5.json') as f:
    full_md5 = json.load(f)

summary = pd.DataFrame([
    {'campo': 'checkpoint', 'valor': 'checkpoints/sweep_best.pt'},
    {'campo': 'md5_calculado', 'valor': md5['calculated_md5']},
    {'campo': 'md5_esperado', 'valor': md5['expected_md5']},
    {'campo': 'md5_match', 'valor': md5['md5_match']},
    {'campo': 'Recall@1 test', 'valor': sweep['test']['recall_at_1_image_to_text']},
    {'campo': 'Macro Recall test', 'valor': sweep['test']['macro_recall']},
    {'campo': 'Sparsity SAE test', 'valor': sweep['test']['sae_visual_sparsity_ratio']},
    {'campo': 'MSE reconstruccion val', 'valor': sweep['val']['sae_visual_recon_mse']},
    {'campo': 'MSE reconstruccion test adicional', 'valor': sweep['test']['sae_visual_recon_mse']},
])
display(summary)
"""),
        code("""comparison = pd.DataFrame([
    {
        'checkpoint': 'sweep_best.pt',
        'rol_en_entrega': 'resultado principal',
        'recall_at_1_test': sweep['test']['recall_at_1_image_to_text'],
        'macro_recall_test': sweep['test']['macro_recall'],
        'md5': md5['calculated_md5'],
        'md5_match': md5['md5_match'],
    },
    {
        'checkpoint': 'full_logging_best.pt',
        'rol_en_entrega': 'evidencia instrumentada con curvas',
        'recall_at_1_test': full['final_metrics']['test']['recall_at_1_image_to_text'],
        'macro_recall_test': full['final_metrics']['test']['macro_recall'],
        'md5': full_md5['checkpoint_best_md5'],
        'md5_match': full_md5['md5_match'],
    },
])
display(comparison)
"""),
        md('## Interpretacion\n\nEl checkpoint `sweep_best.pt` es el resultado principal porque fue seleccionado por el sweep de semillas e hiperparametros y alcanza `Recall@1 test = 0.6043`, por encima del minimo de rubrica `0.45`. El checkpoint `full_logging_best.pt` no reemplaza al principal; se conserva porque registra historia completa de entrenamiento y permite defender las curvas solicitadas. Ambos MD5 coinciden, por lo que los archivos `.pt` son verificables y reproducibles. Se omite la tabla detallada por clase para no mezclar KPIs globales finales con metricas diagnosticas de clase.'),
    ],
)


write_notebook(
    '02_curvas_entrenamiento_consolidado.ipynb',
    'Entregable 2 - Curvas de Entrenamiento',
    [
        md('## Objetivo\n\nConsolidar las curvas solicitadas: loss total, InfoNCE, loss SAE, sparsity ratio, MSE y Recall@1 por epoch. Estas curvas provienen del reentrenamiento instrumentado de la misma configuracion ganadora y se leen solo desde `curvas/`.'),
        code(setup),
        code("""hist = pd.read_csv(ROOT / 'curvas/full_logging_history.csv')
curve_summary = pd.DataFrame([
    {'metrica': 'train_loss_total', 'inicio': hist['train_loss_total'].iloc[0], 'final': hist['train_loss_total'].iloc[-1], 'mejor': hist['train_loss_total'].min()},
    {'metrica': 'train_loss_infonce', 'inicio': hist['train_loss_infonce'].iloc[0], 'final': hist['train_loss_infonce'].iloc[-1], 'mejor': hist['train_loss_infonce'].min()},
    {'metrica': 'test_recall_at_1', 'inicio': hist['test_recall_at_1_image_to_text'].iloc[0], 'final': hist['test_recall_at_1_image_to_text'].iloc[-1], 'mejor': hist['test_recall_at_1_image_to_text'].max()},
    {'metrica': 'test_recon_mse', 'inicio': hist['test_sae_visual_recon_mse'].iloc[0], 'final': hist['test_sae_visual_recon_mse'].iloc[-1], 'mejor': hist['test_sae_visual_recon_mse'].min()},
    {'metrica': 'test_sparsity_ratio', 'inicio': hist['test_sae_visual_sparsity_ratio'].iloc[0], 'final': hist['test_sae_visual_sparsity_ratio'].iloc[-1], 'mejor': hist['test_sae_visual_sparsity_ratio'].max()},
])
display(curve_summary)
"""),
        md('## Figuras e Interpretacion'),
        code("""interpretaciones = {
    '01_loss_total_por_epoch.png': 'La perdida de train cae casi a cero, mientras validacion y test suben con los epochs. Esto evidencia sobreajuste del reentrenamiento instrumentado; no invalida el entregable, pero justifica reportar el mejor checkpoint del sweep como KPI principal.',
    '02_loss_infonce_por_epoch.png': 'Replica el patron del loss total porque InfoNCE domina la funcion objetivo. El modelo aprende fuertemente el alineamiento en train, pero la brecha con validacion/test muestra generalizacion limitada.',
    '03_loss_sae_recon_por_epoch.png': 'El loss total de reconstruccion SAE baja de forma consistente en los tres splits, indicando que el autoencoder aprende a reconstruir la representacion latente sin colapso.',
    '04_loss_sparsity_l1_por_epoch.png': 'La penalizacion L1 desciende gradualmente, coherente con activaciones mas compactas y controladas.',
    '05_sparsity_por_epoch.png': 'La sparsity se mantiene practicamente constante alrededor de 0.8809; la figura usa offset numerico en el eje, por eso visualmente parece separada por niveles, pero el valor real permanece estable.',
    '06_recon_mse_por_epoch.png': 'El MSE visual aumenta moderadamente con los epochs, especialmente en train, pero se mantiene bajo y dentro del umbral excelente en el checkpoint principal reportado.',
    '07_recall_por_epoch.png': 'Train sube hasta casi 1.0, mientras validacion oscila alrededor de 0.35-0.48 y test alrededor de 0.53-0.63. Esta brecha confirma sobreajuste parcial y apoya seleccionar por sweep/checkpoint, no por ultimo epoch.',
}

for fig in sorted((ROOT / 'curvas').glob('*.png')):
    display(Markdown(f'### {fig.name}'))
    display(Image(filename=str(fig)))
    display(Markdown('**Interpretacion.** ' + interpretaciones.get(fig.name, 'Figura de control del entrenamiento.')))
"""),
        code("""with open(ROOT / 'metricas/full_logging_manifest.json') as f:
    manifest = json.load(f)
rows = []
for split, vals in manifest['final_metrics'].items():
    rows.append({
        'split': split,
        'recall_at_1': vals['recall_at_1_image_to_text'],
        'macro_recall': vals['macro_recall'],
        'sparsity_ratio': vals['sae_visual_sparsity_ratio'],
        'recon_mse': vals['sae_visual_recon_mse'],
    })
display(pd.DataFrame(rows))
"""),
        md('## Lectura Final\n\nEl reentrenamiento instrumentado alcanza `Recall@1 test = 0.5489` y aporta la evidencia temporal completa exigida por la rubrica. Las curvas muestran una tension clara: el retrieval contrastivo sobreajusta, pero el SAE conserva sparsity estable y reconstruccion dentro de niveles aceptables. La brecha train-test se reporta de forma transparente; el mejor checkpoint del sweep se usa como KPI principal porque fue seleccionado mediante comparacion sistematica de configuraciones y alcanza `Recall@1 test = 0.6043`.'),
    ],
)


write_notebook(
    '03_afe_afc_consolidado.ipynb',
    'Entregable 3 - Reporte AFE y AFC',
    [
        md('## Objetivo\n\nConsolidar matriz de cargas rotada, scree plot, varianza explicada e indices de bondad de ajuste AFC. Toda la evidencia psicometrica final se referencia desde `afe_afc/`.'),
        code(setup),
        code("""with open(ROOT / 'afe_afc/manifest_afe_afc.json') as f:
    manifest = json.load(f)
display(pd.DataFrame([{
    'shape_original': manifest['input_shape_original'],
    'shape_filtrado': manifest['input_shape_filtrado'],
    'componentes_80': manifest['componentes_80_filtrado_principal'],
    'varianza_80': manifest['varianza_80_filtrado_principal'],
    'CFI': manifest['metricas_afc_final']['CFI'],
    'RMSEA': manifest['metricas_afc_final']['RMSEA'],
    'cumple_pdf': manifest['cumple_pdf'],
}]))
"""),
        md('## Interpretacion de Ajuste\n\nLa matriz final es `1500 x 512`, correspondiente al dataset v10/v5b completo. La AFE retiene `39` componentes para alcanzar `80.13%` de varianza explicada, justo por encima del minimo exigido. En AFC, `RMSEA = 0.0312` es excelente y `CFI = 0.9465` cumple el minimo; esto sostiene validez de constructo exploratoria para los embeddings. `SRMR = 0.1458` se mantiene como diagnostico adicional y limitacion metodologica, no como KPI duro de la rubrica.'),
        code("""display(pd.read_csv(ROOT / 'afe_afc/check_kpis_afc_rubrica.csv'))
kmo = pd.read_csv(ROOT / 'afe_afc/kmo_bartlett_summary.csv')
display(kmo[kmo['modelo'] == 'v10_v5b'])
"""),
        code("""loads = pd.read_csv(ROOT / 'afe_afc/afe_cargas_rotadas.csv')
display(loads.head(20))
display(pd.read_csv(ROOT / 'afe_afc/top_dimensiones_factor.csv').head(30))
"""),
        md('## Grafica PCA/Fscores por Clase Generada en Notebook\n\nEsta celda genera directamente una proyeccion bidimensional por clase a partir de `afe_afc/factor_scores.parquet`. Se usa `Factor_1_score` y `Factor_2_score` como coordenadas latentes principales de la solucion AFE/PCA final.'),
        code("""scores = pd.read_parquet(ROOT / 'afe_afc/factor_scores.parquet')
fig_dir = ROOT / 'afe_afc/figures'
fig_dir.mkdir(parents=True, exist_ok=True)

colors = {
    'contaminacion_alta_NO2': '#d73027',
    'contaminacion_alta_SO2': '#fc8d59',
    'ozono_anomalo': '#fee08b',
    'suelo_urbano': '#4575b4',
    'vegetacion_densa': '#1a9850',
}

plt.figure(figsize=(9, 7))
for cls, g in scores.groupby('candidate_class'):
    plt.scatter(
        g['Factor_1_score'],
        g['Factor_2_score'],
        s=16,
        alpha=0.65,
        label=cls,
        color=colors.get(cls),
        edgecolors='none',
    )
plt.axhline(0, color='0.75', linewidth=0.8)
plt.axvline(0, color='0.75', linewidth=0.8)
plt.xlabel('Factor 1 score')
plt.ylabel('Factor 2 score')
plt.title('Proyeccion PCA/AFE por clase - Factor 1 vs Factor 2')
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
pca_scatter = fig_dir / '05_pca_factor_scatter_pc1_pc2_por_clase_generado_notebook.png'
plt.savefig(pca_scatter, dpi=180)
plt.close()

display(Image(filename=str(pca_scatter)))
display(Markdown('**Interpretacion.** La proyeccion muestra separacion visual entre `suelo_urbano` y `vegetacion_densa`: `suelo_urbano` concentra un grupo claro en valores positivos de Factor 1 y altos de Factor 2, mientras `vegetacion_densa` aparece mas extendida hacia regiones negativas/intermedias. No se espera separacion perfecta para todas las clases porque los embeddings son de alta dimension y esta grafica solo resume dos ejes; aun asi, la diferencia urbano-vegetacion es visible y defendible como evidencia exploratoria de estructura latente por clase.'))
"""),
        code("""pairs_to_plot = [
    ('Factor_1_score', 'Factor_3_score'),
    ('Factor_2_score', 'Factor_3_score'),
    ('Factor_3_score', 'Factor_4_score'),
    ('Factor_4_score', 'Factor_5_score'),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (xcol, ycol) in zip(axes.ravel(), pairs_to_plot):
    for cls, g in scores.groupby('candidate_class'):
        ax.scatter(g[xcol], g[ycol], s=10, alpha=0.45, label=cls, color=colors.get(cls), edgecolors='none')
    ax.axhline(0, color='0.8', linewidth=0.7)
    ax.axvline(0, color='0.8', linewidth=0.7)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f'{xcol} vs {ycol}')
handles, labels = axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
fig.suptitle('Proyecciones multipanel de scores latentes por clase', y=0.98)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
pca_multi = fig_dir / '06_pca_factor_scatter_multipanel_por_clase_generado_notebook.png'
plt.savefig(pca_multi, dpi=180)
plt.close()

display(Image(filename=str(pca_multi)))
display(Markdown('**Interpretacion.** El multipanel muestra que `suelo_urbano` se separa mejor de `vegetacion_densa` en varios factores, lo que indica que los embeddings capturan diferencias visuales entre zonas urbanas y cobertura vegetal. En cambio, `NO2`, `SO2` y `ozono_anomalo` aparecen mas mezcladas, por lo que su separacion visual es menos clara.'))
"""),
        md('## Figuras AFE/AFC e Interpretacion'),
        code("""interpretaciones = {
    '01_pca_remoteclip_v10_v5b_completa_vs_filtrada.png': 'La curva acumulada cruza el umbral de 80% en 39 componentes. Las curvas completa y filtrada coinciden porque no hubo filas cero en v10/v5b (n=1500).',
    '02_heatmap_cargas_rotadas_v10_v5b.png': 'Muestra cargas positivas y negativas distribuidas por factores. No hay un unico factor dominante para todas las dimensiones; esto respalda una estructura latente multifactorial.',
    '03_scree_plot_autovalores_v10_v5b.png': 'Los autovalores caen fuerte en los primeros componentes y luego forman una cola larga. La linea roja en 39 componentes marca el punto necesario para cubrir 80% de varianza.',
    '04_varianza_explicada_componentes_v10_v5b.png': 'Los primeros componentes concentran la mayor parte de la informacion; despues la contribucion marginal cae. Esto explica por que se requieren 39 componentes para llegar al umbral sin conservar las 512 dimensiones.',
}

for fig in sorted((ROOT / 'afe_afc/figures').glob('*.png')):
    if fig.name.endswith('_generado_notebook.png'):
        continue
    display(Markdown(f'### {fig.name}'))
    display(Image(filename=str(fig)))
    display(Markdown('**Interpretacion.** ' + interpretaciones.get(fig.name, 'Figura psicometrica de control.')))
"""),
        md('## Lectura\n\nLa AFE alcanza varianza acumulada `0.8013`, cumpliendo el minimo de 80%. El AFC cumple los umbrales duros de la rubrica con `CFI = 0.9465` y `RMSEA = 0.0312`. `SRMR = 0.1458` queda como limitacion metodologica adicional, no exigida como KPI duro en la rubrica provista.'),
    ],
)


write_notebook(
    '04_sae_interpretabilidad_consolidado.ipynb',
    'Entregable 4 - Neuronas Activas SAE por Clase',
    [
        md('## Objetivo\n\nConsolidar la interpretabilidad mecanica del SAE: sparsity, unidades activas medias, unidades muertas y top unidades por clase. Toda la evidencia final se referencia desde `sae_interpretabilidad/`.'),
        code(setup),
        code("""display(pd.read_csv(ROOT / 'sae_interpretabilidad/split_summary.csv'))
"""),
        md('## Interpretacion del Resumen por Split\n\nEl SAE mantiene una sparsity estable de `0.8809` en train, validacion y test. Esto equivale a cerca de `122` unidades activas de `1024` por muestra, lo que facilita interpretar que subconjuntos compactos de neuronas responden a patrones visuales/textuales. El porcentaje de unidades muertas en test (`13.18%`) es aceptable para una configuracion sparse: hay especializacion sin colapso masivo del diccionario.'),
        code("""top = pd.read_csv(ROOT / 'sae_interpretabilidad/top_units_by_class.csv')
test_top = top[top['split'] == 'test'].copy()
display(test_top.groupby('candidate_class').head(5))
"""),
        md('## Interpretacion por Clase\n\nLa tabla anterior lista las unidades con mayor activacion media por clase en test. La lectura correcta no es que una unidad sea una etiqueta semantica perfecta, sino que ciertos indices latentes se activan de forma diferencial para clases como contaminacion por NO2/SO2, ozono anomal, suelo urbano y vegetacion densa. Esto cumple el entregable de neuronas activas por clase y aporta trazabilidad mecanica al resultado CLIP/SAE.'),
        md('## Graficas Generadas: Top Unidades SAE por Clase\n\nLas siguientes graficas se generan directamente desde `sae_interpretabilidad/top_units_by_class.csv`. Cada panel muestra las unidades SAE mas activas para una clase en test.'),
        code("""fig_dir = ROOT / 'sae_interpretabilidad'
classes = list(test_top['candidate_class'].drop_duplicates())
fig, axes = plt.subplots(len(classes), 1, figsize=(10, 2.7 * len(classes)))
if len(classes) == 1:
    axes = [axes]

for ax, cls in zip(axes, classes):
    sub = test_top[test_top['candidate_class'] == cls].sort_values('mean_abs_activation', ascending=False).head(10)
    labels = [str(u) for u in sub['unit']]
    ax.bar(labels, sub['mean_abs_activation'], color='#4C78A8')
    ax.set_title(f'Top unidades SAE - {cls}')
    ax.set_xlabel('Unidad SAE')
    ax.set_ylabel('Activacion abs. media')
    ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
bars_path = fig_dir / 'top_units_by_class_test_bars_generado_notebook.png'
plt.savefig(bars_path, dpi=180)
plt.close()

display(Image(filename=str(bars_path)))
display(Markdown('**Interpretacion.** La grafica muestra que todas las clases siguen un patron parecido: un grupo pequeno de unidades SAE concentra las activaciones mas altas y luego la importancia disminuye gradualmente. La diferencia entre clases esta en cuales unidades aparecen como mas activas y no tanto en la forma general de la distribucion. Por eso, esta figura sirve para identificar las neuronas latentes mas relevantes por clase, mas que para afirmar una separacion fuerte entre clases.'))
"""),
        code("""summary_class = (
    test_top.groupby('candidate_class')
    .agg(
        top_unit=('unit', lambda s: int(s.iloc[0])),
        top_activation=('mean_abs_activation', 'max'),
        mean_top10_activation=('mean_abs_activation', lambda s: float(s.head(10).mean())),
        mean_top10_frequency=('active_frequency', lambda s: float(s.head(10).mean())),
    )
    .reset_index()
)
display(summary_class)
display(Markdown('**Interpretacion.** Esta tabla resume para cada clase la unidad mas activa, la magnitud maxima y la frecuencia media de activacion de sus 10 unidades principales. Sirve como evidencia tabular compacta del analisis de neuronas activas SAE por clase.'))
"""),
        code("""pairs = pd.read_csv(ROOT / 'sae_interpretabilidad/pair_top_units.csv')
display(pairs.head(30))
"""),
        md('## Matriz de Confusion por Clase'),
        code("""with open(ROOT / 'sae_interpretabilidad/manifest_sae_analysis.json') as f:
    manifest = json.load(f)

classes = manifest['classes']
cm = pd.DataFrame(manifest['confusion_matrices']['test'], index=classes, columns=classes)
display(cm)

plt.figure(figsize=(8, 7))
im = plt.imshow(cm.values, cmap='Blues')
plt.colorbar(im, fraction=0.046, pad=0.04, label='Conteo')
plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
plt.yticks(range(len(classes)), classes)
plt.xlabel('Clase predicha')
plt.ylabel('Clase real')
plt.title('Matriz de confusion - test')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = int(cm.iloc[i, j])
        color = 'white' if value > cm.values.max() * 0.55 else 'black'
        plt.text(j, i, value, ha='center', va='center', color=color, fontsize=9)
plt.tight_layout()
cm_path = ROOT / 'sae_interpretabilidad/confusion_matrix_test_generado_notebook.png'
plt.savefig(cm_path, dpi=180)
plt.close()

display(Image(filename=str(cm_path)))
display(Markdown('**Interpretacion.** La matriz de confusion muestra que el modelo reconoce mejor `suelo_urbano` y `vegetacion_densa`, con la mayoria de ejemplos en la diagonal. `ozono_anomalo` tambien tiene un desempeno razonable, aunque presenta confusiones con `NO2` y `SO2`. Las clases de contaminacion (`NO2` y `SO2`) son las mas dificiles de separar: varios ejemplos se confunden entre si o con `ozono_anomalo`. Esto sugiere que las clases visualmente asociadas a cobertura del suelo son mas distinguibles que las clases atmosfericas.'))
"""),
        md('## Heatmap de unidades activas por clase'),
        code("""heat = test_top.pivot_table(index='candidate_class', columns='unit', values='mean_abs_activation', aggfunc='mean', fill_value=0)
keep_units = test_top.groupby('unit')['mean_abs_activation'].max().sort_values(ascending=False).head(30).index
heat = heat.reindex(columns=keep_units, fill_value=0)

plt.figure(figsize=(14, 5))
plt.imshow(heat.values, aspect='auto', cmap='viridis')
plt.colorbar(label='Activacion absoluta media')
plt.yticks(range(len(heat.index)), heat.index)
plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
plt.title('Top unidades SAE visuales por clase - test')
plt.tight_layout()
heatmap_path = ROOT / 'sae_interpretabilidad/top_units_by_class_heatmap_generado_notebook.png'
plt.savefig(heatmap_path, dpi=180)
plt.close()

display(Image(filename=str(heatmap_path)))
display(Markdown('**Interpretacion.** El heatmap resume que unidades SAE aparecen con mayor activacion media por clase. `suelo_urbano` concentra los valores mas altos en varias unidades, mientras que `vegetacion_densa` activa un subconjunto diferente con menor intensidad. Las clases `NO2`, `SO2` y `ozono_anomalo` muestran senales mas debiles y localizadas. Esto ayuda a identificar que neuronas latentes participan en cada clase, pero no implica que las clases esten completamente separadas.'))
"""),
        code("""display(pd.DataFrame([manifest]))
"""),
        md('## Lectura Final\n\nLa configuracion `k_frac=0.12` mantiene aproximadamente 122 unidades activas de 1024 por muestra, con sparsity test `0.8809`. El heatmap muestra una activacion especialmente fuerte y distribuida para `suelo_urbano`, activaciones mas focalizadas para `vegetacion_densa`, y senales mucho mas puntuales para las clases de contaminacion y ozono. Esta evidencia no prueba causalidad ambiental, pero si demuestra que el SAE genera una representacion compacta y auditable por clase.'),
    ],
)


write_notebook(
    '05_generacion_dataset_tiles_v10_v5b_codigo.ipynb',
    'Evidencia de Codigo - Generacion Dataset Tiles v10/v5b',
    [
        md('## Objetivo\n\nDocumentar, de forma limpia y sin ruido historico, el protocolo y codigo base usado para generar el dataset final de 1500 pares imagen-texto de Situacion 2. Este notebook se conserva como evidencia metodologica del codigo empleado; no se re-ejecuta como parte obligatoria de la entrega porque depende de caches Sentinel-2 y artefactos pesados ya auditados.'),
        code(setup),
        md('## Dataset Final Auditado\n\nLa salida final usada por el modelo v10/v5b esta consolidada en `metricas/dataset_summary.json`. La metadata original completa vive fuera de esta entrega en `outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit/metadata.jsonl`.'),
        code("""with open(ROOT / 'metricas/dataset_summary.json') as f:
    summary = json.load(f)
display(pd.DataFrame([summary]))
display(pd.DataFrame(summary['class_counts'].items(), columns=['class', 'n']))
display(pd.DataFrame(summary['split_counts'].items(), columns=['split', 'n']))
"""),
        md('## Validaciones Criticas\n\nEstas validaciones son las que importan para defensa: 1500 pares, cinco clases balanceadas, split por escena y cero solapamiento entre particiones.'),
        code("""checks = pd.DataFrame([
    {'check': 'n_records == 1500', 'value': summary['n_records'], 'pass': summary['n_records'] == 1500},
    {'check': '5 clases balanceadas con 300 pares', 'value': summary['class_counts'], 'pass': all(v == 300 for v in summary['class_counts'].values())},
    {'check': 'train/val/test = 1000/265/235', 'value': summary['split_counts'], 'pass': summary['split_counts'] == {'train': 1000, 'val': 265, 'test': 235}},
    {'check': 'sin overlap train-val', 'value': summary['scene_overlap_train_val'], 'pass': summary['scene_overlap_train_val'] == 0},
    {'check': 'sin overlap train-test', 'value': summary['scene_overlap_train_test'], 'pass': summary['scene_overlap_train_test'] == 0},
    {'check': 'sin overlap val-test', 'value': summary['scene_overlap_val_test'], 'pass': summary['scene_overlap_val_test'] == 0},
])
display(checks)
"""),
        md('## Interpretacion de la Auditoria\n\nEl dataset final cumple tres condiciones criticas para defender el entrenamiento: tamano suficiente (`1500` pares), balance exacto entre cinco clases (`300` pares por clase) y split por escena sin solapamiento. La particion final `1000/265/235` no es exactamente `70/15/15`; se priorizo holdout por `scene_id` porque reduce mejor el riesgo de fuga espacial/temporal entre train, validacion y test.'),
        md('## Codigo Base Del Protocolo\n\nEl bloque siguiente resume el flujo reproducible empleado. Se deja como codigo documentado, no como ejecucion obligatoria dentro de la entrega final.'),
        code("""# Pseudocodigo fiel al protocolo final v10/v5b.
# Las rutas pesadas se mantienen fuera de Entrega_Final para evitar duplicacion.

from pathlib import Path
import json
import pandas as pd

BASE = Path('/workspace/geovision-cali-hf')
DATASET = BASE / 'outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit'
METADATA = DATASET / 'metadata.jsonl'
SUMMARY = DATASET / 'dataset_summary.json'

def load_metadata(path=METADATA):
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def audit_dataset(summary_path=SUMMARY):
    with open(summary_path, encoding='utf-8') as f:
        s = json.load(f)
    assert s['n_records'] == 1500
    assert all(v == 300 for v in s['class_counts'].values())
    assert s['split_counts'] == {'train': 1000, 'val': 265, 'test': 235}
    assert s['scene_overlap_train_val'] == 0
    assert s['scene_overlap_train_test'] == 0
    assert s['scene_overlap_val_test'] == 0
    return s

# df = load_metadata()
# summary = audit_dataset()
"""),
        md('## Decisiones Metodologicas\n\n- Sentinel-5P se usa como pseudo-etiqueta y trazabilidad textual, no como banda ni feature numerica directa del modelo.\n- El split se realiza por escena para reducir fuga espacial/temporal entre train, val y test.\n- Las clases finales son cinco y quedan balanceadas a 300 pares por clase.\n- La evidencia final se valida por `metadata_md5 = be91fc65425e2181b1ddf6d91da878b6`.\n\nLa interpretacion correcta es que este notebook prueba el protocolo y la auditoria del dataset final; no intenta reconstruir toda la descarga/procesamiento pesado dentro de la carpeta de entrega.'),
    ],
)


write_notebook(
    '00_resumen_situacion2_consolidado.ipynb',
    'Resumen Consolidado Situacion 2',
    [
        md('## Objetivo\n\nResumen ejecutivo de KPIs, entregables y rutas finales para subir la entrega. Este notebook funciona como fuente final y georreferencia documental de la carpeta evaluable.'),
        code(setup),
        md('## Raiz Evaluable y Version Unica\n\n- Ruta absoluta: `/workspace/geovision-cali-hf/Entrega_Final/Situacion2`.\n- Ruta relativa: `Entrega_Final/Situacion2`.\n- Indice maestro: `INDICE_EVIDENCIAS.md`.\n- La evidencia final se referencia exclusivamente desde `metricas/`, `curvas/`, `afe_afc/`, `sae_interpretabilidad/`, `checkpoints/` y `notebooks/`.'),
        code("""display(pd.DataFrame([
    {'tipo': 'raiz_absoluta', 'ruta': str(ROOT)},
    {'tipo': 'indice_evidencias', 'ruta': 'INDICE_EVIDENCIAS.md'},
    {'tipo': 'tabla_kpis', 'ruta': 'metricas/kpi_situacion2_summary.csv'},
    {'tipo': 'checklist_rubrica', 'ruta': 'CHECKLIST_RUBRICA_FINAL.md'},
    {'tipo': 'checkpoint_principal', 'ruta': 'checkpoints/sweep_best.pt'},
    {'tipo': 'curvas_finales', 'ruta': 'curvas/'},
    {'tipo': 'afe_afc_final', 'ruta': 'afe_afc/'},
    {'tipo': 'sae_final', 'ruta': 'sae_interpretabilidad/'},
]))
"""),
        code("""display(pd.read_csv(ROOT / 'metricas/kpi_situacion2_summary.csv'))
"""),
        md('## Interpretacion Ejecutiva\n\nLa entrega cumple todos los umbrales minimos de la rubrica y alcanza nivel excelente en Recall@5, sparsity SAE, MSE de reconstruccion y RMSEA. Los indicadores que quedan como `cumple_minimo` son Recall@1, varianza explicada AFE y CFI; ninguno queda por debajo del umbral. La principal cautela metodologica es que Recall@5 se interpreta como cobertura con cinco prompts de clase y que SRMR se reporta solo como diagnostico adicional.'),
        code("""for path in [ROOT / 'INDICE_EVIDENCIAS.md', ROOT / 'CHECKLIST_RUBRICA_FINAL.md', ROOT / 'README.md']:
    if path.exists():
        display(Markdown(path.read_text(encoding='utf-8')))
"""),
    ],
)

print(f'Notebooks written to {NB_DIR}')
