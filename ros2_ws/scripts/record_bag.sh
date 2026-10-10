#!/usr/bin/env bash
# Grabacion de un bag de camara y microfono con protocolo fijo de ocho etapas.
# Aplicar el mismo guion a todas las personas mantiene comparables las condiciones
# de captura entre bags.
#
# Uso (dentro del contenedor):
#   record_bag.sh <nombre>             graba en /data/bags/<nombre>
#   SEGMENT_S=1 record_bag.sh prueba   variante corta para verificar el script
#
# Se graba unicamente audio crudo: los topics del VAD son derivados, y guardarlos
# congelaria la decision del umbral vigente al grabar. Con /audio/raw en el bag,
# vad_node se puede reejecutar con cualquier configuracion sobre el mismo audio.
#
# Levanta camera_node y audio_capture_node, graba y detiene ambos al terminar.

source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

set -euo pipefail

NAME="${1:?Uso: record_bag.sh <nombre>}"
OUT_DIR="/data/bags/${NAME}"
TOPICS=(/camera/image_raw /audio/raw)
SEGMENT_S="${SEGMENT_S:-5}"

CUES=(
  "Silencio, de frente a la camara"
  "Di tu nombre y de donde eres"
  "Sigue hablando mientras giras la cabeza a los lados"
  "Haz tu pregunta al sistema"
  "Callado: acercate y alejate de la camara"
  "Silencio, y a la mitad tose o golpea la mesa"
  "Luz baja, hablando"
  "Luz fuerte apuntando a la cara, hablando"
)
DURATION=$(( SEGMENT_S * ${#CUES[@]} ))

# ros2 bag record no sobrescribe carpetas existentes; se valida antes de abrir los
# dispositivos.
if [ -e "$OUT_DIR" ]; then
  echo "Ya existe $OUT_DIR. Usa otro nombre o borra esa carpeta." >&2
  exit 1
fi

# Un nodo previo mantendria la camara o el microfono ocupados.
for pattern in camera_node audio_capture_node; do
  if pgrep -f "[${pattern:0:1}]${pattern:1}" > /dev/null; then
    echo "Hay un ${pattern} corriendo. Apagalo antes de grabar." >&2
    exit 1
  fi
done

mkdir -p /data/bags

# Con control de trabajos activo cada proceso en segundo plano queda en su propio
# grupo, y un solo kill al grupo alcanza a ros2 run y al nodo que lanza. Sin set -m,
# bash inicia los procesos en segundo plano con SIGINT ignorado.
set -m
ros2 run hri_camera camera_node > /tmp/record_camera_node.log 2>&1 &
CAMERA_PID=$!
ros2 run hri_audio audio_capture_node > /tmp/record_audio_node.log 2>&1 &
AUDIO_PID=$!

# SIGINT permite a los nodos liberar camara y microfono; SIGKILL de respaldo si algun
# grupo sigue vivo tras 2 s.
cleanup() {
  for pid in "$CAMERA_PID" "$AUDIO_PID"; do
    kill -INT -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "$CAMERA_PID" "$AUDIO_PID"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# Verifica que cada topic publique antes de grabar. Se reintenta porque los nodos
# tardan unos segundos en arrancar y anunciar sus publicadores.
wait_for_topic() {
  local topic="$1" log="$2" last_err=""
  for _ in $(seq 1 20); do
    if last_err=$(timeout 3 ros2 topic echo --once --no-arr "$topic" 2>&1 > /dev/null); then
      return 0
    fi
    sleep 1
  done
  echo "No llegan mensajes en $topic tras 20 intentos." >&2
  echo "Ultimo error de ros2 topic echo: ${last_err}" >&2
  echo "Log del nodo (${log}):" >&2
  tail -n 10 "$log" >&2
  return 1
}

wait_for_topic /camera/image_raw /tmp/record_camera_node.log
wait_for_topic /audio/raw /tmp/record_audio_node.log

echo "Guion de ${DURATION} s (${SEGMENT_S} s por etapa). Grabando '${NAME}' en:"
echo "  ${OUT_DIR}"
for n in 5 4 3 2 1; do
  echo "Empieza en ${n}..."
  sleep 1
done

(
  for i in "${!CUES[@]}"; do
    echo ">>> [$(( i * SEGMENT_S ))-$(( (i + 1) * SEGMENT_S )) s] ${CUES[$i]}"
    sleep "$SEGMENT_S"
  done
) &
CUES_PID=$!

# timeout envia SIGINT, que ros2 bag record interpreta como cierre ordenado del
# archivo. Los 2 s adicionales absorben la latencia de arranque del grabador.
timeout -s INT $(( DURATION + 2 )) \
  ros2 bag record --topics "${TOPICS[@]}" -o "$OUT_DIR" --storage-preset-profile zstd_fast || true

wait "$CUES_PID" 2>/dev/null || true
echo "Listo. Resumen del bag:"
ros2 bag info "$OUT_DIR"
