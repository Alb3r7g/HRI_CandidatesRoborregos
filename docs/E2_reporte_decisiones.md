# E2 · Reporte de decisiones

**Reto HRI/Visión, Candidates RoBorregos 2026-2**
Albert Constantino Salazar · octubre de 2026

El sistema decide quién habla frente a la laptop, verifica su identidad fusionando cara,
voz y movimiento de boca, y solo entonces responde con el documento privado de esa
persona. Este reporte explica cómo está dividido en ROS 2, de dónde salen los umbrales,
cómo se calibró la confianza fusionada y qué no funcionó.

## 1. Arquitectura ROS 2

El sistema son 9 nodos repartidos en 6 paquetes, más `hri_interfaces` (mensajes y servicio) y `hri_bringup` (launch y umbrales), y un servidor MCP que corre como subproceso.
Todos se levantan con un solo `ros2 launch`.

```
camera_node ──/camera/image_raw──► face_detection_node ──/vision/tracked_faces──► identity_node
                                                                                     │ /vision/identified_faces
audio_capture_node ──/audio/raw──► vad_node ──/audio/speech_segment──► transcription_node ──► /audio/transcript ──┐
                                                    └──────────────────► voice_identity_node ──► /audio/voice_identity ──┤
                                                                                                                         ▼
                                                     rag_service_node ◄──── servicio /rag/query ──── fusion_node
                                                        │ MCP (stdio)                                    │
                                                     mcp_server ─► ChromaDB          /fusion/overlay_image, /hri/dialog
```

### Qué es tópico, qué es servicio y por qué

**Tópicos: todo lo que es un flujo continuo de datos.** Cuadros (~29 Hz), bloques de
audio (~31 Hz), caras rastreadas e identificadas, actividad de voz, frases,
transcripciones y voces. Se eligieron tópicos por tres razones:

- **No hay respuesta que esperar.** El productor no debe bloquearse si un consumidor va
  lento.
- **Hay varios consumidores.** `/audio/speech_segment` lo leen a la vez la transcripción y
  la verificación de voz. `/camera/image_raw` lo leen la detección, la identidad y la
  fusión.
- **Se pueden grabar y reproducir.** Los bags guardan solo los tópicos crudos
  (`/camera/image_raw`, `/audio/raw`). Así todo lo derivado se recalcula con otros
  umbrales sobre los mismos datos.

**Servicio: la consulta al documento, `/rag/query` (`QueryDocument.srv`).**

- **Es petición-respuesta.** Cada pregunta necesita exactamente una respuesta y un
  estado: `ANSWERED`, `NOT_IN_DOCUMENT`, `DENIED` o `LLM_UNAVAILABLE`.
- **La identidad va dentro de la petición.** `person` e `identity_confidence` son campos
  obligatorios, así que no existe forma de consultar sin identidad. Así se implementa "la
  identidad es un permiso, no una llave de ruteo".
- **El servidor vuelve a validar la confianza** contra su propio umbral (0.85) antes de
  tocar el índice o el LLM. Si la consulta se niega, devuelve siempre el mismo texto, sin
  importar la razón, para no enseñarle a un atacante qué le falló.

**Acciones: no se usaron.** Una acción sirve para metas largas con retroalimentación
parcial y cancelación. La única operación larga en línea es la consulta al RAG, que tarda
1–7 s. No tiene progreso intermedio útil: el LLM devuelve un JSON completo que hay que
verificar entero antes de mostrarlo, así que no hay respuesta parcial que enseñar. Una
acción agregaría complejidad sin beneficio. El candidato natural a acción es el
enrolamiento (retroalimentación "muestra N de M", cancelable). Hoy es un script fuera del grafo de ROS,
porque se corre una vez por persona y con el sistema apagado.

### Interfaces propias

`hri_interfaces` define 10 mensajes y 1 servicio. **`AudioChunk` y `SpeechSegment` llevan
su formato** (`sample_rate`, `channels`), así que un bag es autodescriptivo. **Los `stamp`
marcan cuándo ocurrió el evento, no cuándo llegó**, para que la fusión alinee cada frase con
los cuadros del mismo intervalo. **`DialogTurn`** publica cada turno completo (pregunta,
cara, voz, boca, confianza, decisión, razón y respuesta), así que cada decisión es auditable.

**La lógica de fusión vive en `fusion.py`, sin ROS.** El nodo en vivo y la evaluación
offline usan el mismo código, así que la curva FAR/FRR mide lo que corre en la demo. **La
boca se mide dentro de `identity_node`**, donde el modelo de 68 puntos ya corre sobre el
mismo recorte. Su valor viaja en un campo propio (`mouth_open`).

## 2. Umbrales de identidad facial y de voz

### Datos

Grabamos 9 bags con un guion fijo de 40 s (8 etapas de 5 s), solo con los tópicos
crudos:

| Tipo | Bags | Contenido |
|---|---|---|
| Genuino | Albert_02, Albert_Tablet_01, Albert_Tablet_02 | de frente, con tablet, lejos y de lado |
| Desconocido | Desconocido_01, Desconocido_02 | dos personas no enroladas haciendo las mismas preguntas |
| Grupo | Grupo_01 | 3 personas en cuadro, una habla por etapa |
| Ataque | Ataque_Foto, Ataque_Video, Ataque_Audio | selfie en un celular; video reproducido; voz grabada con la boca cerrada |

### Cara (InsightFace `buffalo_l`, ArcFace, coseno)

Medido cada tercer cuadro de los bags:

- **Genuino:** 0.47–0.79. El peor caso es a ~1.5 m y de lado.
- **Impostores:** como máximo 0.16 en **1,846 caras**.
- **Separación:** con cualquier umbral entre 0.40 y 0.45, FAR = 0 % y FRR = 0 % cuadro a
  cuadro. El umbral original de 0.50 rechazaba el 0.5 % de los cuadros genuinos.

### Voz (ECAPA-TDNN de SpeechBrain, 192 dimensiones, coseno)

- **3 frases de un bag** (dejando una fuera): genuino 0.51–0.55, impostores ≤ 0.39. Margen
  muy delgado.
- **Enrolamiento en vivo de 60 s:** genuino 0.59–0.64, impostores ≤ 0.34.
- **Con todo el sistema corriendo**, la voz cayó a 0.38–0.52 por el ventilador de la
  laptop: efecto de canal.
- **Huella final:** promedio de un enrolamiento en silencio y otro con ruido. Genuino
  0.46–0.65, impostores ≤ 0.36.

### Decisión: el nodo nombra, la fusión decide

Los umbrales de los nodos son bajos a propósito: **cara 0.20 y voz 0.35**, iguales al
ancla inferior de cada señal (sección 3). No deciden el acceso; solo marcan desde dónde una
señal cuenta para la fusión. Si `identity_node` exigiera 0.50, Albert a 1.5 m (0.48)
llegaría como "desconocido" y la fusión no podría pedirle que se acerque.

## 3. Confianza fusionada

### Paso 1: quién habla

La decisión se toma por frase, nunca cuadro a cuadro:

1. Para cada frase que corta el VAD se toman las caras vistas en su intervalo (±0.3 s),
   agrupadas por `track_id`.
2. **El hablante** es el track con mayor actividad de boca: la desviación estándar de la
   apertura de los labios, normalizada por el ancho de la boca para que no dependa de la
   distancia.
3. **Su identidad** es el nombre más frecuente de ese track en la ventana, y **su cara**
   es la mediana de su similitud. La mediana es robusta a cuadros malos.
4. **La voz solo cuenta** si ECAPA atribuye la frase a esa misma persona.

### Paso 2: normalización por anclas

`norm(x) = clip((x − bajo) / (alto − bajo), 0, 1)`. Las anclas salen de los datos: "bajo"
es lo máximo que dan los impostores y "alto" es lo mínimo de un genuino claro.

| Señal | Bajo (vale 0) | Alto (vale 1) | Datos |
|---|---|---|---|
| Cara | 0.20 | 0.55 | impostores ≤ 0.16; genuino ≥ 0.47 |
| Voz | 0.35 | 0.60 | impostores ≤ 0.36; genuino 0.46–0.65 |
| Boca | 0.006 | 0.020 | callado 0.006–0.009; hablando ≥ 0.020 (AUC 0.75) |

### Paso 3: suma ponderada

**C = 0.5·cara + 0.3·voz + 0.2·boca**

- **La cara pesa más** porque es la señal más separada (≥ 0.47 contra ≤ 0.16).
- **La voz pesa menos** porque es la más sensible al canal (ruido, distancia, frases
  cortas).
- **La boca aporta poco a la identidad,** pero es la evidencia de que hay alguien vivo
  hablando.

Se eligió la suma ponderada sobre un modelo aprendido porque con 35 intentos etiquetados
un clasificador se sobreajustaría, y porque cada decisión se descompone en sus tres términos
(lo que muestran `DialogTurn` y el overlay).

**Consecuencia medible:** cualquier persona que hable aporta 0.2 por la boca. Por eso
todos los desconocidos dan exactamente **C = 0.20**, y el corte bajo tiene que estar
claramente arriba de ese valor.

### Paso 4: compuerta de vida

La suma pura no frenaba los ataques: la foto daba C = 0.54–0.59 y el audio grabado
0.66–0.70, porque la cara o la voz eran reales y la boca pesa solo 0.2. Se agregó una
regla dura: **si la actividad de boca del hablante es < 0.015, el puntaje es 0**. Con
ella, la foto, el audio grabado y el ruido de fondo caen a 0.

### Paso 5: política de tres estados

| Situación | Decisión |
|---|---|
| C ≥ 0.90 | **Responder** y conceder el permiso al track |
| Ya tiene permiso y C ≥ 0.85 (histéresis, margen 0.05) | **Responder** |
| 0.65 ≤ C < 0.90, o boca quieta, por primera vez | **Preguntar**: "Por seguridad. Mira a la cámara y repite tu pregunta." |
| C < 0.65, o segunda vez en la franja (en 20 s) | **Negar** y revocar |

### Paso 6: el permiso se vigila

Cada 0.1 s se revisa a quien tiene permiso, y se revoca si su cara deja de verse más de
2 s, si su confianza continua queda debajo de 0.85 más de 1 s o si empieza a hablar otro
track.

**El permiso está atado al `track_id` que lo ganó, no al nombre.** Lo descubrimos en una
prueba de punta a punta: un video reproducido justo después de la sesión de Albert
heredaba su acceso (0.87 ≥ 0.85) porque se reconocía como "Albert". Atarlo al track lo
corrigió.

## 4. Curva FAR vs FRR y punto de operación

Los 9 bags se reprocesaron con `evaluate_fusion.py`, que usa la misma lógica que el nodo.
El resultado son **35 intentos etiquetados: 13 genuinos y 22 impostores** (11 de
desconocidos, 2 de otras personas del grupo y 9 ataques). Una frase más era ruido de
arranque y se excluyó.

<img src="img/far_frr.png" alt="FAR y FRR por umbral, y cada intento" width="75%">

| Umbral sobre C | FRR (genuinos rechazados) | FAR (impostores aceptados) |
|---|---|---|
| 0.21 – 0.65 | 1/13 = 7.7 % | 3/22 = 13.6 % |
| 0.70 | 2/13 = 15.4 % | 3/22 = 13.6 % |
| 0.80 | 3/13 = 23.1 % | 3/22 = 13.6 % |
| 0.85 | 3/13 = 23.1 % | 1/22 = 4.5 % |
| **0.90** | **5/13 = 38.5 %** | **0/22 = 0 %** |

Los 3 impostores que sobreviven hasta 0.85 son **los tres videos reproducidos**
(C = 0.80–0.865). El video trae la cara, la voz y una boca que se mueve, así que pasa la
compuerta de vida. Desconocidos, fotos y audio grabado quedan en 0.20 o en 0.

**Punto de operación: corte alto = 0.90.** Es el umbral más bajo con FAR = 0 %. En un
sistema que da acceso a información privada, un falso aceptado es mucho más caro que un
falso rechazado.

**El costo es un FRR de 38.5 % en respuesta directa, y la re-pregunta lo absorbe.**

- **De dónde salen los genuinos bajo 0.90:** 3 de los 5 son del bag lejos y de lado
  (C = 0.86–0.87), donde la cara baja a 0.48–0.52.
- **Por qué los rescata la re-pregunta:** su texto ("mira a la cámara") corrige justo lo
  que los bajó.
- **Corte bajo = 0.65:** deja en "preguntar" a todos los genuinos salvo uno y está muy
  arriba del 0.20 de los desconocidos. **Solo 1 de 13 genuinos (7.7 %) se niega de
  plano.**

**Segunda puerta en el RAG: 0.85**, no 0.90: con la histéresis, quien ya tiene permiso lo
conserva hasta 0.85, y un RAG más estricto contradiría a la fusión.

**Lo que no se cumplió:** el objetivo inicial era FRR ≤ 5 % sumando respuesta y
re-pregunta. No se alcanza sin bajar el corte bajo a 0.20, donde entraría cualquier
desconocido. Se prefirió la seguridad.

**Límite de la muestra:** 35 intentos de una sola persona enrolada dan una curva
indicativa, no una estimación con intervalos estrechos. Además, la tabla es de antes del
enrolamiento de voz combinado.

## 5. RAG con la identidad como permiso

- **Una colección de ChromaDB por persona.** Buscar en el documento de otra persona no es
  un filtro que se pueda olvidar: es otra tabla.
- **Embeddings `BAAI/bge-m3`, elegido midiendo** contra multilingual-e5-large y MiniLM
  (10/10, 10/10 y 8/10). Ganó por tener el rango de similitud más ancho y por no necesitar
  prefijos.
- **MCP:** la búsqueda es la herramienta `search_document` de un servidor MCP por stdio.
  Su único cliente es el servicio, que valida la confianza antes de llamarla.
- **LLM local:** `qwen2.5:7b` en Ollama con `temperature = 0`. Devuelve JSON con una
  respuesta y **citas literales**, y el código verifica que cada cita exista en el
  documento. Si no se verifica, la respuesta se reemplaza por "no encontré esa
  información".

| Persona | Recuperación (top-3) | Respuestas correctas | Trampas rechazadas | Mediana |
|---|---|---|---|---|
| Albert | 10/10 | 10/10 | 3/3 | 2.0 s |
| Sebastian | 7/7 | 7/7 | 3/3 | 1.4 s |

Las trampas incluyen un dato ausente cuyo tema sí aparece en el documento, un dato
declarado privado y **un dato del documento de la otra persona**. Este último es la
prueba directa de que cada quien recibe respuesta solo de su propio documento.

## 6. Qué se probó y falló

1. **Rechazar preguntas sin respuesta con un umbral de similitud.** En los tres modelos de
   embeddings, el rango de las preguntas contestables se solapa con el de las trampas
   (margen −0.086 con bge-m3): una pregunta sobre el hotel trae el fragmento del viaje
   porque el tema está aunque el dato no. El rechazo tiene que venir del LLM, con citas
   verificadas.
2. **Verificación de citas exacta y citas cortas.** qwen copia mal ~1 de cada 4 citas
   largas, así que la verificación exacta rechazaba respuestas correctas. Se volvió
   tolerante (≥ 80 % de las palabras en tramos de ≥ 3 seguidas). Pedir citas de ≤ 20
   palabras bajó el resultado a 6/10 y se revirtió.
3. **Confusión de persona en el documento de Sebastian.** El documento describía a su
   roomie Albert en tercera persona con frases parecidas a las suyas ("Estudia
   Ingeniería…"). El modelo contestó la carrera de Albert y le atribuyó el atletismo
   (6/7). La verificación de citas detectó la cita fabricada. Redactar nombrando al dueño
   de cada dato lo llevó a 7/7.
4. **Suma ponderada sin compuerta de vida.** No frenaba la foto ni el audio grabado
   (sección 3, paso 4).
5. **El video reproducido pasa la compuerta de vida** (C = 0.80–0.865). Lo frena el corte
   de 0.90 y, si insiste, la re-pregunta y la negación. Un anti-spoofing activo de
   verdad necesitaría un reto aleatorio ("di este número").
6. **Permiso atado al nombre.** El video heredaba el acceso de la sesión anterior. Se ató
   al `track_id`.
7. **El VAD junta hablantes en grupo.** Con 750 ms de silencio para cerrar una frase,
   dos personas que hablan seguido quedan en un solo segmento: la voz se mezcla y bajó
   de 0.49 a 0.26. Es el único genuino negado de plano (C = 0.20). Hay una tensión real:
   la transcripción quiere frases largas y la verificación de voz quiere un solo
   hablante.
8. **Efecto de canal en la voz.** La primera prueba en vivo dio acceso en 1 de 7 frases por
   el ventilador; con el enrolamiento combinado, 13 de 17.
9. **Otros límites conocidos:** las frases cortas dan poca voz (0.38–0.43); Whisper
   alucina con ruido ("Amara.org"); MediaPipe se descartó porque instala numpy 2, que rompe
   `cv_bridge`; la webcam USB se caía del bus y se reemplazó por una tablet con IP Webcam;
   la primera frase tras arrancar puede salir "boca quieta" mientras cargan los modelos
   (falla segura: vuelve a preguntar).
