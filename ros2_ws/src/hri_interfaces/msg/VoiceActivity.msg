# msg/VoiceActivity.msg
# Actividad de voz por bloque de audio, al ritmo de /audio/raw. Se publica la
# probabilidad cruda además de la decisión booleana para poder recalibrar el
# umbral sobre datos ya grabados y para alimentar el valor de confianza del
# overlay. header.stamp se hereda del AudioChunk que la originó, de modo que
# la señal queda alineada con los frames de cámara.
std_msgs/Header header
float32 speech_probability
bool is_speech
