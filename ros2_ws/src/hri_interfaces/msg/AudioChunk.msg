# msg/AudioChunk.msg
# El formato viaja junto a las muestras en vez de en un topic aparte, para que
# cualquier consumidor (VAD, transcripción, verificación de voz) pueda
# interpretar un mensaje suelto y para que un bag quede autodescriptivo.
# stamp marca el inicio del bloque, no su llegada: es el instante en que
# ocurrió el sonido y es lo que permite alinearlo con los frames de cámara.
std_msgs/Header header
float32[] samples
uint32 sample_rate
uint8 channels
