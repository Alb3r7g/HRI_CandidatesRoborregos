#!/usr/bin/env bash
# Grabacion de un bag de /camera/image_raw con protocolo fijo de seis etapas.
# Aplicar el mismo guion a todas las personas mantiene comparables las condiciones
# de captura entre bags.
#
# Uso (dentro del contenedor):
#   record_bag.sh <nombre>             graba en /data/bags/<nombre>
#   SEGMENT_S=1 record_bag.sh prueba   variante corta para verificar el script
#
# Levanta camera_node, graba unicamente el topic de la camara y detiene el nodo
# al terminar.

source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash

set -euo pipefail

NAME="${1:?Uso: record_bag.sh <nombre>}"
OUT_DIR="/data/bags/${NAME}"
TOPIC="/camera/image_raw"
SEGMENT_S="${SEGMENT_S:-5}"

CUES=(
  "De frente, luz normal"
  "Gira la cabeza suavemente a los lados (unos 30 grados)"
  "Acercate y alejate de la camara"
  "Habla o mueve los labios"
  "Luz baja"
  "Luz fuerte apuntando a la cara"
)
DURATION=$(( SEGMENT_S * ${#CUES[@]} ))

# ros2 bag record no sobrescribe carpetas existentes; se valida antes de abrir la camara.
if [ -e "$OUT_DIR" ]; then
  echo "Ya existe $OUT_DIR. Usa otro nombre o borra esa carpeta." >&2
  exit 1
fi

# Un camera_node previo mantendria /dev/video0 ocupado.
if pgrep -f "[c]amera_node" > /dev/null; then
  echo "Hay un camera_node corriendo. Apagalo antes de grabar." >&2
  exit 1
fi

mkdir -p /data/bags

# Con control de trabajos activo cada proceso en segundo plano queda en su propio
# grupo, y un solo kill al grupo alcanza a ros2 run y al nodo que lanza. Sin set -m,
# bash inicia los procesos en segundo plano con SIGINT ignorado.
set -m
ros2 run hri_camera camera_node > /tmp/record_camera_node.log 2>&1 &
CAMERA_PID=$!

# SIGINT permite a camera_node liberar la camara; SIGKILL de respaldo si el grupo
# sigue vivo tras 2 s.
cleanup() {
  kill -INT -- "-$CAMERA_PID" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$CAMERA_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Verifica que el topic publique frames antes de grabar. Se reintenta porque
# camera_node tarda unos segundos en arrancar y anunciar el publicador.
LAST_ERR=""
GOT_FRAME=0
for attempt in $(seq 1 20); do
  if LAST_ERR=$(timeout 3 ros2 topic echo --once --no-arr "$TOPIC" 2>&1 > /dev/null); then
    GOT_FRAME=1
    break
  fi
  sleep 1
done
if [ "$GOT_FRAME" -ne 1 ]; then
  echo "No llegan frames en $TOPIC tras 20 intentos." >&2
  echo "Ultimo error de ros2 topic echo: ${LAST_ERR}" >&2
  echo "Log de camera_node (/tmp/record_camera_node.log):" >&2
  tail -n 10 /tmp/record_camera_node.log >&2
  exit 1
fi

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
  ros2 bag record --topics "$TOPIC" -o "$OUT_DIR" --storage-preset-profile zstd_fast || true

wait "$CUES_PID" 2>/dev/null || true
echo "Listo. Resumen del bag:"
ros2 bag info "$OUT_DIR"
