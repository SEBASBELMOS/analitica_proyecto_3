# Limitación NO2 Para Validación Espacial

Con los insumos DAGMA/SISAIRE disponibles para Situación 3, la cobertura por contaminante es:

| Contaminante | Estaciones disponibles |
|---|---:|
| NO2 | 1 |
| SO2 | 5 |
| O3 | 7 |

Esto implica que `NO2` no permite una validación `leave-one-out` espacial defendible. Si se oculta la única estación de `NO2`, no queda ninguna estación del mismo contaminante para entrenar o ajustar el kriging residual.

Tratamiento metodológico propuesto:

| Contaminante | Tratamiento |
|---|---|
| O3 | LOO-CV espacial real + métricas por horizonte |
| SO2 | LOO-CV espacial real + métricas por horizonte |
| NO2 | Modelado como salida, validación temporal puntual si aplica, sin KPI LOO-CV espacial |

Frase para reporte:

```text
NO2 se modela como salida del sistema, pero no se reporta LOO-CV espacial para NO2 por insuficiencia de estaciones DAGMA/SISAIRE disponibles (n=1). Los mapas de NO2 deben interpretarse como extrapolación no validable espacialmente con los datos actuales y con incertidumbre elevada.
```

Esta limitación es de disponibilidad observacional, no del pipeline computacional.
